// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// One configuration per binary; no device-side variant dispatch. The build
// manifest and Python module expose the same name. Default preserves control.
#ifndef GDN_SM90_CONFIGURATION
#define GDN_SM90_CONFIGURATION 0
#endif
namespace gdn::sm90 {
enum class Configuration { Control, Value64, Value64LocalInverse, Value128Paired };
inline constexpr auto ConfigurationId = Configuration(GDN_SM90_CONFIGURATION);
static_assert(GDN_SM90_CONFIGURATION >= 0 && GDN_SM90_CONFIGURATION <= 3,
              "unknown SM90 configuration");
struct ConfigurationTraits {
    static constexpr bool Tuned = ConfigurationId != Configuration::Control;
    static constexpr bool LocalInverse = ConfigurationId == Configuration::Value64LocalInverse;
    static constexpr bool PackedNewV = ConfigurationId == Configuration::Value128Paired;
    static constexpr bool PairedTail = LocalInverse || PackedNewV;
    static constexpr int ValueTile = PackedNewV || !Tuned ? 128 : 64;
    static constexpr int AuxLoadBudget = Tuned ? 128 : 176;
    static constexpr int AuxRegisters = !Tuned ? 152 : ValueTile == 64 ? 232 : 104;
    static constexpr const char* Name = !Tuned ? "control" :
        LocalInverse ? "value64-local-inverse" :
        PackedNewV ? "value128-paired" : "value64";
};
} // namespace gdn::sm90
