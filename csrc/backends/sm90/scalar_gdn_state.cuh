// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "scalar_gdn_aux.cuh"
#include "configuration.cuh"
#include "fragment_convert.cuh"

namespace gdn::sm90 {

// Scalar-gated state recurrence. No temporary Q/K scaling tile: scalar gates
// commute with QH/KH and with K^T delta. This is not a vector-KDA operation.
// The inherited state/inverse storage precision and producer/consumer counts
// remain fixed; old KDA mainloop is retained independently.
template<class Base, bool AuxInverse = false>
struct ScalarGdnState : ScalarGdnAux<Base,AuxInverse> {
    using Parent = ScalarGdnAux<Base,AuxInverse>;
    using Element = typename Base::Element;
    using Inverse = typename Base::InverseType;
    using Params = typename Base::Params;
    using SharedStorage = typename Parent::SharedStorage;
    using Value = cute::Int<Base::ValueTile>;

    template<class Problem, class Work>
    CUTE_DEVICE void compute(
        Params const& params, Problem const& problem, Work const& work,
        typename Base::MainloopQPipeline& qp, typename Base::QPipelineState& qr,
        typename Base::MainloopKPipeline& kp, typename Base::KPipelineState& kr,
        typename Base::MainloopVPipeline& vp, typename Base::VPipelineState& vr,
        typename Base::MainloopOPipeline& op, typename Base::OPipelineState& ow,
        typename Base::MainloopQKPipeline& qkp, typename Base::QKPipelineState& qkr,
        typename Base::MainloopKKPipeline& kkp, typename Base::KKPipelineState& kkr,
        typename Base::MainloopAlphaPipeline& ap, typename Base::AlphaPipelineState& ar,
        typename Base::MainloopBetaPipeline& bp, typename Base::BetaPipelineState& br,
        typename Base::MainloopAlphaLastPipeline& alp, typename Base::AlphaLastPipelineState& alr,
        typename Parent::OrderedMathBarriers& order, SharedStorage& smem) {
        using namespace cute;
        using kda::sm90::collective::gemm_zero_acc;
        using Barriers = kda::sm90::collective::KdaNamedBarriers;
        int tid = int(threadIdx.x) - Base::NumLoadThreads;
        int wg = tid / 128;
        int local_tid = tid % 128;
        auto q = make_tensor(make_smem_ptr(smem.smem_q.data()), typename Base::QKSmemLayoutQ{});
        auto k = make_tensor(make_smem_ptr(smem.smem_k.data()), typename Base::QKSmemLayoutK{});
        auto kt = make_tensor(make_smem_ptr(smem.smem_k.data()), typename Base::KVSmemLayoutK{});
        auto v = make_tensor(make_smem_ptr(smem.smem_v.data()), typename Base::KVSmemLayoutV{});
        auto qk = make_tensor(make_smem_ptr(smem.smem_qk.data()), typename Base::SmemLayoutQK{});
        auto kk = make_tensor(make_smem_ptr(smem.smem_kk.data()), typename Base::SmemLayoutKK{});
        auto kk_operand = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem.smem_kk.data())),
                                      typename Base::SmemLayoutKK{});
        auto o = make_tensor(make_smem_ptr(smem.smem_o.data()), typename Base::SmemLayoutO{});
        auto alpha = make_tensor(make_smem_ptr(smem.smem_alpha.data()), typename Base::QKQSmemLayoutAlpha{});
        auto beta = make_tensor(make_smem_ptr(smem.smem_beta.data()), typename Base::SmemLayoutBeta{});

        auto kv_mma = typename Base::TiledMmaKV{};
        auto o1_mma = typename Base::TiledMmaO1{};
        auto o2_mma = typename Base::TiledMmaO2{};
        auto sk_mma = typename Base::TiledMmaSK{};
        auto newv_mma = typename Base::TiledMmaNewV{};
        auto inv_mma = typename Base::TiledMmaKK{};
        auto kv_thread = kv_mma.get_thread_slice(tid);
        auto o1_thread = o1_mma.get_thread_slice(tid);
        auto o2_thread = o2_mma.get_thread_slice(tid);
        auto sk_thread = sk_mma.get_thread_slice(tid);
        auto newv_thread = newv_mma.get_thread_slice(tid);
        auto inv_thread = inv_mma.get_thread_slice(local_tid);
        auto h = partition_fragment_C(kv_thread, Shape<Value,_128>{});
        auto q_desc = o1_thread.make_fragment_B(o1_thread.partition_B(q));
        auto k_desc = sk_thread.make_fragment_B(sk_thread.partition_B(k));
        auto kt_desc = kv_thread.make_fragment_B(kv_thread.partition_B(kt));
        auto qk_desc = o2_thread.make_fragment_B(o2_thread.partition_B(qk));
        auto inv_desc = newv_thread.make_fragment_B(newv_thread.partition_B(kk_operand));
        auto c_output = o1_thread.partition_C(make_identity_tensor(Shape<Value,_64>{}));
        auto c_value = kv_thread.partition_A(make_identity_tensor(Shape<Value,_64>{}));
        auto c_inverse = inv_thread.partition_C(make_identity_tensor(Shape<_64,_64>{}));

        auto load_v = make_tiled_copy_C(Copy_Atom<SM75_U16x8_LDSM_T,Element>{}, sk_mma);
        auto lv = load_v.get_thread_slice(tid);
        auto store_o = make_tiled_copy_C(typename Base::CollectiveStoreO::CopyAtomR2S{}, o1_mma);
        auto so = store_o.get_thread_slice(tid);
        auto copy_h = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy,float>{}, kv_mma);
        auto ch = copy_h.get_thread_slice(tid);
        if constexpr (Base::kInitStateFromInput) {
            auto global_h = make_tensor(make_gmem_ptr(params.ptr_input_state),
                state_layout<128,128>(problem.num_v_heads,problem.num_seqs))(
                    _,_,work.o_head_idx(),work.seq_idx);
            if constexpr (Base::ValueTile == 128) {
                copy(copy_h, ch.partition_S(kda::sm90::collective::select_tensor<1,0>(global_h)), h);
            } else {
                auto half = local_tile(domain_offset(make_coord(_0{},work.value_offset),global_h),
                                       Shape<_128,Value>{},make_coord(_0{},_0{}));
                copy(copy_h,ch.partition_S(kda::sm90::collective::select_tensor<1,0>(half)),h);
            }
        } else {
            clear(h);
        }

        auto inverse = [&]() __attribute__((always_inline)) {
            auto slice = kk(_,_,kkr.index());
            typename Base::CollectiveInverse solve(Barriers::StateMathWG0);
            solve.compute(slice);
            cutlass::arch::NamedBarrier::arrive_and_wait(128,Barriers::StateMathWG0);
            auto ld = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N,Inverse>{}, inv_mma);
            auto st = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N,Element>{}, inv_mma);
            auto l = ld.get_thread_slice(local_tid);
            auto s = st.get_thread_slice(local_tid);
            auto inverse_fp16 = make_fragment_like<Inverse>(partition_fragment_C(inv_thread,Shape<_64,_64>{}));
            auto operand_bf16 = make_fragment_like<Element>(inverse_fp16);
            copy(ld,l.partition_S(slice),l.retile_D(inverse_fp16));
            CUTE_UNROLL
            for (int i=0; i<size(inverse_fp16); ++i) {
                auto [row,col] = c_inverse(i);
                operand_bf16(i) = Element(float(inverse_fp16(i))*beta(col,br.index()));
            }
            copy(st,s.retile_S(operand_bf16),s.partition_D(kk_operand(_,_,kkr.index())));
        };

        auto body = [&](int chunk, auto first_tag, auto last_tag) __attribute__((always_inline)) {
            constexpr bool first = decltype(first_tag)::value;
            constexpr bool last = decltype(last_tag)::value;
            int valid = last ? int(work.seq_len-chunk*64) : 64;
            ap.consumer_wait(ar);
            qp.consumer_wait(qr);
            auto acc_o = partition_fragment_C(o1_thread,Shape<Value,_64>{});
            if constexpr (!first) {
                auto operand_h = kda::sm90::collective::make_acc_into_op<Element>(h,typename Base::TiledMmaO1::LayoutA_TV{});
                warpgroup_fence_operand(operand_h);
                warpgroup_fence_operand(acc_o);
                order.ordered_or_wait(wg);
                warpgroup_arrive();
                gemm_zero_acc(o1_mma,operand_h,q_desc(_,_,_,qr.index()),acc_o);
                warpgroup_commit_batch(); order.notify_next_blocked(wg);
                warpgroup_wait<0>(); warpgroup_fence_operand(acc_o);
                CUTE_UNROLL
                for (int i=0; i<size(acc_o); ++i) {
                    auto [dv,t] = c_output(i);
                    acc_o(i) *= smem.gate_factors[ar.index()*128+64+t];
                }
            }
            qp.consumer_release(qr); ++qr;

            kp.consumer_wait(kr);
            auto acc_sk = partition_fragment_C(sk_thread,Shape<Value,_64>{});
            if constexpr (!first) {
                auto operand_h = kda::sm90::collective::make_acc_into_op<Element>(h,typename Base::TiledMmaSK::LayoutA_TV{});
                warpgroup_fence_operand(operand_h);
                warpgroup_fence_operand(acc_sk);
                order.ordered_or_wait(wg);
                warpgroup_arrive();
                gemm_zero_acc(sk_mma,operand_h,k_desc(_,_,_,kr.index()),acc_sk);
                warpgroup_commit_batch(); order.notify_next_blocked(wg);
                warpgroup_wait<0>(); warpgroup_fence_operand(acc_sk);
            }
            vp.consumer_wait(vr);
            auto residual = make_fragment_like<Element>(acc_sk);
            copy(load_v,lv.partition_S(v)(_,_,_,vr.index()),lv.retile_D(residual));
            if constexpr (!first) {
                CUTE_UNROLL
                for (int i=0; i<size(residual); ++i) {
                    auto [dv,t] = c_output(i);
                    residual(i) = residual(i)-Element(acc_sk(i)*smem.gate_factors[ar.index()*128+t]);
                }
            }
            kkp.consumer_wait(kkr);
            bp.consumer_wait(br);
            if constexpr (!AuxInverse) {
                if (wg==0) inverse();
                cutlass::arch::NamedBarrier::arrive_and_wait(256,Barriers::StateMath);
                cutlass::arch::fence_view_async_shared();
            }

            auto acc_delta = partition_fragment_C(newv_thread,Shape<Value,_64>{});
            {
                auto operand_r = kda::sm90::collective::make_acc_into_op<Element>(residual,typename Base::TiledMmaNewV::LayoutA_TV{});
                warpgroup_fence_operand(operand_r);
                warpgroup_fence_operand(acc_delta);
                order.ordered_or_wait(wg);
                warpgroup_arrive();
                gemm_zero_acc(newv_mma,operand_r,inv_desc(_,_,_,kkr.index()),acc_delta);
                warpgroup_commit_batch(); order.notify_next_blocked(wg);
                warpgroup_wait<0>(); warpgroup_fence_operand(acc_delta);
            }
            vp.consumer_release(vr); ++vr;
            kkp.consumer_release(kkr); ++kkr;
            bp.consumer_release(br); ++br;

            auto operand_delta = [&] {
                if constexpr (ConfigurationTraits::PackedNewV) {
                    auto delta_bf16 = convert_fragment<Element>(acc_delta);
                    return kda::sm90::collective::make_acc_into_op<Element>(delta_bf16,typename Base::TiledMmaKV::LayoutA_TV{});
                } else {
                    return kda::sm90::collective::make_acc_into_op<Element>(acc_delta,typename Base::TiledMmaKV::LayoutA_TV{});
                }
            }();
            qkp.consumer_wait(qkr);
            warpgroup_fence_operand(operand_delta);
            warpgroup_fence_operand(acc_o);
            order.ordered_or_wait(wg);
            warpgroup_arrive();
            if constexpr (first) gemm_zero_acc(o2_mma,operand_delta,qk_desc(_,_,_,qkr.index()),acc_o);
            else gemm(o2_mma,operand_delta,qk_desc(_,_,_,qkr.index()),acc_o);
            warpgroup_commit_batch(); order.notify_next_blocked(wg);
            if constexpr (ConfigurationTraits::PairedTail) {
                // Consume the inherited alpha-last pipeline even though the scalar
                // prefix supplies the same value; do not change barrier counts.
                alp.consumer_wait(alr);
                float decay_h = smem.gate_factors[ar.index()*128+valid-1];
                CUTE_UNROLL
                for (int i=0; i<size(h); ++i) h(i) *= decay_h;
                // O2 still owns the unscaled operand. A distinct buffer preserves
                // that async lifetime while scalar work and KV can make progress.
                auto operand_scaled = make_fragment_like<Element>(operand_delta);
                CUTE_UNROLL
                for (int i=0; i<size(operand_delta); ++i) {
                    auto [dv,t] = c_value(i);
                    float gain = smem.relative_gate[relative_gate_index(ar.index(),t)];
                    operand_scaled(i) = Element(float(operand_delta(i))*gain);
                }
                warpgroup_fence_operand(operand_scaled);
                warpgroup_fence_operand(h);
                order.ordered_or_wait(wg);
                warpgroup_arrive();
                gemm(kv_mma,operand_scaled,kt_desc(_,_,_,kr.index()),h);
                warpgroup_commit_batch(); order.notify_next_blocked(wg);
                warpgroup_wait<0>();
                warpgroup_fence_operand(h);
                warpgroup_fence_operand(acc_o);
                qkp.consumer_release(qkr); ++qkr;
                {
                    auto output = make_fragment_like<Element>(acc_o);
                    copy(acc_o,output);
                    op.producer_acquire(ow);
                    copy(store_o,so.retile_S(output),so.partition_D(o(_,_,ow.index())));
                    cutlass::arch::fence_view_async_shared();
                    op.producer_commit(ow); ++ow;
                }
            } else {
                warpgroup_wait<0>(); warpgroup_fence_operand(acc_o);
                qkp.consumer_release(qkr); ++qkr;
                {
                    auto output = make_fragment_like<Element>(acc_o);
                    copy(acc_o,output);
                    op.producer_acquire(ow);
                    copy(store_o,so.retile_S(output),so.partition_D(o(_,_,ow.index())));
                    cutlass::arch::fence_view_async_shared();
                    op.producer_commit(ow); ++ow;
                }

                // Consume the inherited alpha-last pipeline even though the scalar
                // prefix supplies the same value; do not change barrier counts.
                alp.consumer_wait(alr);
                float decay_h = smem.gate_factors[ar.index()*128+valid-1];
                CUTE_UNROLL
                for (int i=0; i<size(h); ++i) h(i) *= decay_h;
                CUTE_UNROLL
                for (int i=0; i<size(operand_delta); ++i) {
                    auto [dv,t] = c_value(i);
                    float gain = smem.relative_gate[relative_gate_index(ar.index(),t)];
                    operand_delta(i) = Element(float(operand_delta(i))*gain);
                }
                warpgroup_fence_operand(operand_delta);
                warpgroup_fence_operand(h);
                order.ordered_or_wait(wg);
                warpgroup_arrive();
                gemm(kv_mma,operand_delta,kt_desc(_,_,_,kr.index()),h);
                warpgroup_commit_batch(); order.notify_next_blocked(wg);
                warpgroup_wait<0>(); warpgroup_fence_operand(h);
            }
            kp.consumer_release(kr); ++kr;
            ap.consumer_release(ar); ++ar;
            alp.consumer_release(alr); ++alr;
        };

        int chunks = ceil_div(work.seq_len,64);
        if (chunks==1) body(0,cute::bool_constant<!Base::kInitStateFromInput>{},cute::true_type{});
        else body(0,cute::bool_constant<!Base::kInitStateFromInput>{},cute::false_type{});
        CUTE_NO_UNROLL
        for (int chunk=1; chunk<chunks-1; ++chunk) body(chunk,cute::false_type{},cute::false_type{});
        if (chunks>1) body(chunks-1,cute::false_type{},cute::true_type{});
        if (params.ptr_output_state) {
            auto global_h = make_tensor(make_gmem_ptr(params.ptr_output_state),
                state_layout<128,128>(problem.num_v_heads,problem.num_seqs))(
                    _,_,work.o_head_idx(),work.seq_idx);
            if constexpr (Base::ValueTile == 128) {
                copy(copy_h,h,ch.partition_D(kda::sm90::collective::select_tensor<1,0>(global_h)));
            } else {
                auto half = local_tile(domain_offset(make_coord(_0{},work.value_offset),global_h),
                                       Shape<_128,Value>{},make_coord(_0{},_0{}));
                copy(copy_h,h,ch.partition_D(kda::sm90::collective::select_tensor<1,0>(half)));
            }
        }
    }
};

} // namespace gdn::sm90
