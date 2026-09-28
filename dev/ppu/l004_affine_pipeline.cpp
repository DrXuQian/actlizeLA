#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

constexpr int kD = 4;
constexpr int kChunk = 3;
using Matrix = std::vector<double>;

struct Inputs {
  int sequence = 0;
  Matrix q;
  Matrix k;
  Matrix v;
  std::vector<double> g;
  std::vector<double> beta;
};

struct Chunk {
  int first = 0;
  int valid = 0;
  Matrix kd;
  Matrix qd;
  Matrix kr;
  Matrix inv;
  Matrix mqk;
  double gt = 1.0;
};

struct Transfer {
  Matrix a;
  Matrix b;
};

struct Options {
  bool wrong_inverse_sign = false;
  bool reverse_scan_composition = false;
  bool inclusive_replay_prefix = false;
};

Matrix zeros(int rows, int columns) {
  return Matrix(std::size_t(rows * columns), 0.0);
}

Matrix identity(int size) {
  Matrix out = zeros(size, size);
  for (int i = 0; i < size; ++i) out[std::size_t(i * size + i)] = 1.0;
  return out;
}

Matrix multiply(
    Matrix const& a, int m, int k, Matrix const& b, int n) {
  Matrix out = zeros(m, n);
  for (int row = 0; row < m; ++row) {
    for (int inner = 0; inner < k; ++inner) {
      double const av = a[std::size_t(row * k + inner)];
      for (int column = 0; column < n; ++column) {
        out[std::size_t(row * n + column)] +=
            av * b[std::size_t(inner * n + column)];
      }
    }
  }
  return out;
}

Transfer compose(Transfer const& later, Transfer const& earlier) {
  Transfer out{multiply(later.a, kD, kD, earlier.a, kD),
               multiply(later.a, kD, kD, earlier.b, kD)};
  for (std::size_t i = 0; i < out.b.size(); ++i) out.b[i] += later.b[i];
  return out;
}

Inputs make_inputs(int sequence) {
  Inputs x{};
  x.sequence = sequence;
  x.q = zeros(sequence, kD);
  x.k = zeros(sequence, kD);
  x.v = zeros(sequence, kD);
  x.g.resize(std::size_t(sequence));
  x.beta.resize(std::size_t(sequence));
  for (int token = 0; token < sequence; ++token) {
    x.g[std::size_t(token)] = -0.004 * (1 + token % 5);
    x.beta[std::size_t(token)] = 0.23 + 0.07 * (token % 4);
    for (int dim = 0; dim < kD; ++dim) {
      x.q[std::size_t(token * kD + dim)] =
          0.018 * (1 + (token * 3 + dim * 5) % 9);
      x.k[std::size_t(token * kD + dim)] =
          0.013 * (-4 + (token * 5 + dim * 2) % 9);
      x.v[std::size_t(token * kD + dim)] =
          0.027 * (-3 + (token * 7 + dim * 3) % 8);
    }
  }
  return x;
}

Chunk prepare_chunk(Inputs const& x, int first, Options const& options) {
  Chunk c{};
  c.first = first;
  c.valid = std::min(kChunk, x.sequence - first);
  c.kd = zeros(kChunk, kD);
  c.qd = zeros(kChunk, kD);
  c.kr = zeros(kChunk, kD);
  c.inv = zeros(kChunk, kChunk);
  c.mqk = zeros(kChunk, kChunk);
  Matrix ki = zeros(kChunk, kD);
  std::array<double, kChunk> gamma{};
  double cumulative = 0.0;
  for (int row = 0; row < c.valid; ++row) {
    cumulative += x.g[std::size_t(first + row)];
    gamma[std::size_t(row)] = cumulative;
  }
  c.gt = std::exp(cumulative);
  double const scale = 1.0 / std::sqrt(double(kD));
  for (int row = 0; row < c.valid; ++row) {
    for (int dim = 0; dim < kD; ++dim) {
      int const source = (first + row) * kD + dim;
      double const kval = x.k[std::size_t(source)];
      double const decay = std::exp(gamma[std::size_t(row)]);
      c.kd[std::size_t(row * kD + dim)] = kval * decay;
      c.qd[std::size_t(row * kD + dim)] =
          x.q[std::size_t(source)] * decay * scale;
      ki[std::size_t(row * kD + dim)] = kval / decay;
      c.kr[std::size_t(row * kD + dim)] =
          kval * std::exp(cumulative - gamma[std::size_t(row)]);
    }
  }
  Matrix ki_t = zeros(kD, kChunk);
  for (int row = 0; row < kChunk; ++row) {
    for (int dim = 0; dim < kD; ++dim) {
      ki_t[std::size_t(dim * kChunk + row)] =
          ki[std::size_t(row * kD + dim)];
    }
  }
  Matrix const l_full = multiply(c.kd, kChunk, kD, ki_t, kChunk);
  Matrix const qk_full = multiply(c.qd, kChunk, kD, ki_t, kChunk);
  for (int row = 0; row < kChunk; ++row) {
    for (int column = 0; column < kChunk; ++column) {
      if (row >= column && row < c.valid && column < c.valid) {
        c.mqk[std::size_t(row * kChunk + column)] =
            qk_full[std::size_t(row * kChunk + column)];
      }
    }
  }
  // Solve (I + L)^{-1} column by column. L is strict-lower and its row is
  // scaled by beta[row], exactly as torch.solve_triangular(unitriangular).
  for (int column = 0; column < kChunk; ++column) {
    for (int row = 0; row < kChunk; ++row) {
      if (row < column) continue;
      if (row == column) {
        c.inv[std::size_t(row * kChunk + column)] = 1.0;
        continue;
      }
      double sum = 0.0;
      for (int inner = column; inner < row; ++inner) {
        double const lower =
            l_full[std::size_t(row * kChunk + inner)] *
            (row < c.valid ? x.beta[std::size_t(first + row)] : 0.0);
        sum += lower * c.inv[std::size_t(inner * kChunk + column)];
      }
      c.inv[std::size_t(row * kChunk + column)] =
          options.wrong_inverse_sign ? sum : -sum;
    }
  }
  return c;
}

Transfer chunk_transfer(
    Inputs const& x, Chunk const& c, Transfer const& incoming) {
  Matrix bkd = zeros(kChunk, kD);
  Matrix bkv = zeros(kChunk, kD);
  for (int row = 0; row < c.valid; ++row) {
    double const beta = x.beta[std::size_t(c.first + row)];
    for (int dim = 0; dim < kD; ++dim) {
      bkd[std::size_t(row * kD + dim)] =
          c.kd[std::size_t(row * kD + dim)] * beta;
      bkv[std::size_t(row * kD + dim)] =
          x.v[std::size_t((c.first + row) * kD + dim)] * beta;
    }
  }
  Matrix const p = multiply(c.inv, kChunk, kChunk, bkd, kD);
  Matrix const r = multiply(c.inv, kChunk, kChunk, bkv, kD);
  Matrix const pa = multiply(p, kChunk, kD, incoming.a, kD);
  Matrix const pb = multiply(p, kChunk, kD, incoming.b, kD);

  // c.kr is [C,D], so form kr^T explicitly for the ordinary host GEMM.
  Matrix kr_t = zeros(kD, kChunk);
  for (int row = 0; row < kChunk; ++row) {
    for (int dim = 0; dim < kD; ++dim) {
      kr_t[std::size_t(dim * kChunk + row)] =
          c.kr[std::size_t(row * kD + dim)];
    }
  }
  Matrix const kr_pa = multiply(kr_t, kD, kChunk, pa, kD);
  Matrix const kr_pb = multiply(kr_t, kD, kChunk, pb, kD);
  Matrix const kr_r = multiply(kr_t, kD, kChunk, r, kD);
  Transfer out{zeros(kD, kD), zeros(kD, kD)};
  for (int i = 0; i < kD * kD; ++i) {
    out.a[std::size_t(i)] = c.gt * incoming.a[std::size_t(i)] -
                            kr_pa[std::size_t(i)];
    out.b[std::size_t(i)] = c.gt * incoming.b[std::size_t(i)] -
                            kr_pb[std::size_t(i)] + kr_r[std::size_t(i)];
  }
  return out;
}

void replay_chunk(
    Inputs const& x, Chunk const& c, Matrix& state, Matrix& output) {
  Matrix const kd_state = multiply(c.kd, kChunk, kD, state, kD);
  Matrix error = zeros(kChunk, kD);
  for (int row = 0; row < c.valid; ++row) {
    double const beta = x.beta[std::size_t(c.first + row)];
    for (int dim = 0; dim < kD; ++dim) {
      error[std::size_t(row * kD + dim)] =
          (x.v[std::size_t((c.first + row) * kD + dim)] -
           kd_state[std::size_t(row * kD + dim)]) *
          beta;
    }
  }
  Matrix const u = multiply(c.inv, kChunk, kChunk, error, kD);
  Matrix const inter = multiply(c.qd, kChunk, kD, state, kD);
  Matrix const intra = multiply(c.mqk, kChunk, kChunk, u, kD);
  for (int row = 0; row < c.valid; ++row) {
    for (int dim = 0; dim < kD; ++dim) {
      output[std::size_t((c.first + row) * kD + dim)] =
          inter[std::size_t(row * kD + dim)] +
          intra[std::size_t(row * kD + dim)];
    }
  }
  Matrix kr_t = zeros(kD, kChunk);
  for (int row = 0; row < kChunk; ++row) {
    for (int dim = 0; dim < kD; ++dim) {
      kr_t[std::size_t(dim * kChunk + row)] =
          c.kr[std::size_t(row * kD + dim)];
    }
  }
  Matrix const update = multiply(kr_t, kD, kChunk, u, kD);
  for (int i = 0; i < kD * kD; ++i) {
    state[std::size_t(i)] = c.gt * state[std::size_t(i)] + update[std::size_t(i)];
  }
}

std::pair<Matrix, Matrix> direct(Inputs const& x) {
  Matrix state = zeros(kD, kD);
  Matrix output = zeros(x.sequence, kD);
  double const scale = 1.0 / std::sqrt(double(kD));
  for (int token = 0; token < x.sequence; ++token) {
    double const decay = std::exp(x.g[std::size_t(token)]);
    for (double& value : state) value *= decay;
    std::array<double, kD> delta{};
    for (int column = 0; column < kD; ++column) {
      double memory = 0.0;
      for (int reduction = 0; reduction < kD; ++reduction) {
        memory += x.k[std::size_t(token * kD + reduction)] *
                  state[std::size_t(reduction * kD + column)];
      }
      delta[std::size_t(column)] =
          (x.v[std::size_t(token * kD + column)] - memory) *
          x.beta[std::size_t(token)];
    }
    for (int row = 0; row < kD; ++row) {
      for (int column = 0; column < kD; ++column) {
        state[std::size_t(row * kD + column)] +=
            x.k[std::size_t(token * kD + row)] * delta[std::size_t(column)];
      }
    }
    for (int column = 0; column < kD; ++column) {
      for (int reduction = 0; reduction < kD; ++reduction) {
        output[std::size_t(token * kD + column)] +=
            scale * x.q[std::size_t(token * kD + reduction)] *
            state[std::size_t(reduction * kD + column)];
      }
    }
  }
  return {output, state};
}

std::pair<Matrix, Matrix> pipeline(
    Inputs const& x, int group_chunks, Options const& options) {
  int const chunk_count = (x.sequence + kChunk - 1) / kChunk;
  int const group_count = (chunk_count + group_chunks - 1) / group_chunks;
  std::vector<Chunk> chunks;
  for (int first = 0; first < x.sequence; first += kChunk) {
    chunks.push_back(prepare_chunk(x, first, options));
  }
  std::vector<Transfer> groups;
  for (int group = 0; group < group_count; ++group) {
    Transfer transfer{identity(kD), zeros(kD, kD)};
    int const begin = group * group_chunks;
    int const end = std::min(chunk_count, begin + group_chunks);
    for (int chunk = begin; chunk < end; ++chunk) {
      transfer = chunk_transfer(x, chunks[std::size_t(chunk)], transfer);
    }
    groups.push_back(std::move(transfer));
  }
  std::vector<Transfer> scan = groups;
  for (int offset = 1; offset < group_count; offset <<= 1) {
    std::vector<Transfer> next = scan;
    for (int group = offset; group < group_count; ++group) {
      next[std::size_t(group)] = options.reverse_scan_composition
          ? compose(scan[std::size_t(group - offset)], scan[std::size_t(group)])
          : compose(scan[std::size_t(group)], scan[std::size_t(group - offset)]);
    }
    scan = std::move(next);
  }

  Matrix output = zeros(x.sequence, kD);
  Matrix final_state = zeros(kD, kD);
  for (int group = 0; group < group_count; ++group) {
    Matrix state = zeros(kD, kD);
    if (options.inclusive_replay_prefix) {
      state = scan[std::size_t(group)].b;
    } else if (group > 0) {
      state = scan[std::size_t(group - 1)].b;
    }
    int const begin = group * group_chunks;
    int const end = std::min(chunk_count, begin + group_chunks);
    for (int chunk = begin; chunk < end; ++chunk) {
      replay_chunk(x, chunks[std::size_t(chunk)], state, output);
    }
    if (group == group_count - 1) final_state = std::move(state);
  }
  return {output, final_state};
}

double max_difference(
    std::pair<Matrix, Matrix> const& a,
    std::pair<Matrix, Matrix> const& b) {
  double maximum = 0.0;
  for (std::size_t i = 0; i < a.first.size(); ++i) {
    maximum = std::max(maximum, std::abs(a.first[i] - b.first[i]));
  }
  for (std::size_t i = 0; i < a.second.size(); ++i) {
    maximum = std::max(maximum, std::abs(a.second[i] - b.second[i]));
  }
  return maximum;
}

}  // namespace

int main() {
  constexpr std::array<int, 11> sequences{1, 2, 3, 4, 5, 6, 7, 10, 17, 25, 65};
  constexpr std::array<int, 5> group_chunks{1, 2, 3, 4, 8};
  int cases = 0;
  double good_max = 0.0;
  for (int sequence : sequences) {
    Inputs const input = make_inputs(sequence);
    auto const reference = direct(input);
    for (int group : group_chunks) {
      good_max = std::max(
          good_max,
          max_difference(reference, pipeline(input, group, Options{})));
      ++cases;
    }
  }

  Inputs const witness = make_inputs(65);
  auto const reference = direct(witness);
  double const bad_sign = max_difference(
      reference, pipeline(witness, 2, Options{true, false, false}));
  double const bad_order = max_difference(
      reference, pipeline(witness, 2, Options{false, true, false}));
  double const bad_prefix = max_difference(
      reference, pipeline(witness, 2, Options{false, false, true}));
  bool const pass = good_max < 1.0e-12 && bad_sign > 1.0e-6 &&
                    bad_order > 1.0e-6 && bad_prefix > 1.0e-6;
  std::printf(
      "[ppu affine pipeline] %s cases=%d direct_max=%.3e "
      "inverse-sign=%.3e/EXPECTED-RED scan-order=%.3e/EXPECTED-RED "
      "inclusive-prefix=%.3e/EXPECTED-RED\n",
      pass ? "PASS" : "FAIL", cases, good_max, bad_sign, bad_order,
      bad_prefix);
  return pass ? 0 : 1;
}
