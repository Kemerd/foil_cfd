// CUDA render-kernel API for the cotton-tuft flow-visualization mode: a grid
// of short flexible strands ("tufts") taped to the suction surface, each a
// chain of nodes that the local lattice velocity drags downstream. Attached
// flow lays a tuft flat and steady; separated flow makes it flutter and
// reverse — exactly how a real wind-tunnel / flight-test tuft survey reveals
// the stall line. The root node is pinned to the foil skin; the free nodes
// advect each frame with a spring-to-rest + damping mini cloth sim, and a
// per-tuft "attachment" scalar (0 = attached, 1 = separated) colors the strand
// green -> red so the stall line reads at a glance.
//
// Layout mirrors the particle system (render/particles.cuh): all GL specifics
// stay in viz.cpp; these kernels only see raw mapped pointers. Positions live
// in LATTICE CELL space so the shared velocity sampler needs no rescale.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

#include "../sim/lbm_core.cuh"

namespace foilcfd {

// ===========================================================================
// Tuft buffer layout (binding contract with the GL side):
//   nodes : float4 per NODE — xyz = node position in LATTICE CELL space,
//           w = the tuft's attachment scalar in [0,1] (0 = attached/green,
//           1 = separated/red; the same value is written to every node of a
//           strand so the whole strand reads one color). Nodes are stored
//           strand-contiguous: tuft t occupies [t*nodesPerTuft, (t+1)*nodes).
//           The renderer draws each strand as a GL_LINE_STRIP over its slice.
//
// The host also keeps a DEVICE-ONLY "anchor" buffer (TuftAnchor below) that
// persists between frames: it pins the root and stores the rest geometry the
// cloth sim relaxes toward. Anchors are rebuilt only when the foil geometry
// changes (AoA / airfoil / VG edit), never per frame.
// ===========================================================================

/// Nodes per tuft strand (root + free nodes). Five gives a strand long enough
/// to visibly curl and reverse in separated flow while staying cheap (a few
/// thousand tufts * 5 nodes is trivial next to 1M particles). Compile-time so
/// the GL side can size its draw slices identically.
inline constexpr int kTuftNodes = 5;

/// @brief Persistent per-tuft anchor (device-only, survives between frames).
/// Built on the host at seeding time from the suction-surface frame, then
/// uploaded once; the advection kernel reads it every frame to pin the root
/// and to know the rest length / surface tangent each strand relaxes toward.
struct TuftAnchor {
    float3 root;     ///< Pinned root position on the foil skin (lattice space).
    float3 normal;   ///< Unit outward surface normal: the direction a tuft
                     ///< STANDS UP along at rest (taped at the root, free end
                     ///< pointing off the skin). The flow then bends it over.
    float3 tangent;  ///< Unit surface tangent (downstream, lattice space): the
                     ///< direction an ATTACHED tuft lies along once the flow has
                     ///< pushed it flat. The attachment metric measures how far
                     ///< the bent strand departs from this.
    float  segLen;   ///< Rest length of each segment (cells). The strand wants
                     ///< node i at root + i*segLen along its bent path.
};

/// @brief Tunables for one tuft advection launch.
struct TuftAdvectParams {
    float dtSteps   = 1.0f;  ///< Sim steps advanced this frame (advection time).
    float damping   = 0.65f; ///< Velocity-blend toward the sampled flow per
                             ///< step (0 = ignore flow, 1 = snap to flow). Lower
                             ///< values give the lazy cotton "flutter".
    float flowScale = 6.0f;  ///< Multiplies sampled velocity when displacing
                             ///< nodes, so a tuft a few cells long visibly
                             ///< streams in the freestream (u_lat ~ 0.08).
    float sampleFloor = 1.25f; ///< Minimum wall distance (cells, along the
                             ///< anchor normal) the flow is SAMPLED at. The
                             ///< halfway bounce-back wall plus the voxel
                             ///< stair-step leave a dead band of ~zero velocity
                             ///< right at the skin; nodes hugging the surface
                             ///< must feel the overlying flow or they lag the
                             ///< free tip and the strand shears into a fold.
                             ///< Node POSITIONS are unaffected — only where
                             ///< their drag velocity is read from.
    float sepScale  = 0.5f;  ///< Reversal/cross-flow magnitude (fraction of the
                             ///< root speed) that maps the attachment scalar to
                             ///< 1.0 (fully separated/red).
};

/// @brief Advance every tuft one frame: pin node 0 to its anchor, drag the
/// free nodes with the trilinearly-sampled velocity field, then re-project
/// each link to its EXACT rest length (hard follow-the-leader constraint, the
/// standard PBD rope/hair scheme). The hard links matter: a soft spring lets
/// the chain stretch when near-wall nodes sit in slow air while the tip rides
/// fast air, and the lagging strand folds back over itself. Inextensible
/// links can only rotate about their parent, so the strand stays a single
/// clean line however hard the shear. Each strand's attachment scalar is
/// written into all of its nodes' .w. Defensive against a not-yet-converged
/// solver: non-finite samples leave the node at rest instead of poisoning the
/// VBO.
/// @param nodes    Mapped GL VBO: float4 per node (see layout above).
/// @param anchors  Device anchor buffer (tuftCount entries), persistent.
/// @param tuftCount Number of strands.
/// @param vel      Velocity field view (lattice units).
/// @param params   Launch tunables.
/// @param stream   Stream to run on (the app's single stream).
/// @return cudaGetLastError() from the launch.
cudaError_t launchTuftAdvect(float4* nodes, const TuftAnchor* anchors,
                             int tuftCount, DeviceVelocityField vel,
                             const TuftAdvectParams& params,
                             cudaStream_t stream);

/// @brief Reset every strand to its straight rest pose along the surface
/// tangent (root + i*segLen*tangent) with attachment scalar 0. Run once after
/// (re)seeding the anchors so a freshly placed grid draws sensibly on the very
/// first frame, before any advection has bent it.
/// @param nodes     Mapped GL VBO to fill (tuftCount * kTuftNodes float4).
/// @param anchors   Device anchor buffer (tuftCount entries).
/// @param tuftCount Number of strands.
/// @param stream    Stream to run on.
/// @return cudaGetLastError() from the launch.
cudaError_t launchTuftReset(float4* nodes, const TuftAnchor* anchors,
                            int tuftCount, cudaStream_t stream);

} // namespace foilcfd
