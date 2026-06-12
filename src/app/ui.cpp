// ImGui panel implementation: backend init/teardown, the FoilCFD dark style,
// the default docking layout, and every plan-9.2 panel (Airfoil — including
// the plan-15.5 searchable Aircraft section — / VG editor / VG guidance /
// Sim / Readouts / View) plus the plan-7.2 STL import modal.
// All edits land in UIParams; all sim-affecting actions raise UIEvents that
// main.cpp applies after rendering — the panels never touch the solver.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "ui.h"

#include <imgui.h>
#include <imgui_internal.h> // DockBuilder API (docking branch)
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace foilcfd {

namespace {

// ===========================================================================
// Style: dark, minimal, generous spacing. One typography scale (the default
// font at 1.0 with size-tiered headers via separators rather than font swaps
// — keeps the atlas to a single font and the panels visually quiet).
// ===========================================================================

/// @brief Apply the FoilCFD theme on top of StyleColorsDark: deep neutral
/// backgrounds, a single restrained blue accent, soft rounding, and roomy
/// padding so dense engineering readouts stay scannable.
void applyFoilStyle() {
    ImGui::StyleColorsDark();
    ImGuiStyle& s = ImGui::GetStyle();

    // -- geometry: generous spacing, soft corners --
    s.WindowPadding    = ImVec2(14, 12);
    s.FramePadding     = ImVec2(8, 5);
    s.ItemSpacing      = ImVec2(10, 7);
    s.ItemInnerSpacing = ImVec2(8, 5);
    s.IndentSpacing    = 16.0f;
    s.ScrollbarSize    = 12.0f;
    s.GrabMinSize      = 10.0f;
    s.WindowRounding   = 6.0f;
    s.FrameRounding    = 4.0f;
    s.GrabRounding     = 4.0f;
    s.PopupRounding    = 6.0f;
    s.TabRounding      = 4.0f;
    s.WindowBorderSize = 0.0f;
    s.FrameBorderSize  = 0.0f;

    // -- palette: near-black surfaces, desaturated text, one blue accent --
    ImVec4* c = s.Colors;
    const ImVec4 bg0(0.075f, 0.080f, 0.090f, 1.00f); // window
    const ImVec4 bg1(0.110f, 0.115f, 0.130f, 1.00f); // frames
    const ImVec4 bg2(0.160f, 0.170f, 0.190f, 1.00f); // hovered frames
    const ImVec4 acc(0.26f, 0.55f, 0.96f, 1.00f);    // accent blue
    const ImVec4 accDim(0.20f, 0.38f, 0.62f, 1.00f);
    c[ImGuiCol_WindowBg]          = ImVec4(bg0.x, bg0.y, bg0.z, 0.97f);
    c[ImGuiCol_ChildBg]           = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_PopupBg]           = ImVec4(bg0.x, bg0.y, bg0.z, 0.98f);
    c[ImGuiCol_FrameBg]           = bg1;
    c[ImGuiCol_FrameBgHovered]    = bg2;
    c[ImGuiCol_FrameBgActive]     = ImVec4(0.20f, 0.22f, 0.26f, 1.0f);
    c[ImGuiCol_TitleBg]           = bg0;
    c[ImGuiCol_TitleBgActive]     = ImVec4(0.10f, 0.11f, 0.13f, 1.0f);
    c[ImGuiCol_Header]            = ImVec4(accDim.x, accDim.y, accDim.z, 0.35f);
    c[ImGuiCol_HeaderHovered]     = ImVec4(accDim.x, accDim.y, accDim.z, 0.55f);
    c[ImGuiCol_HeaderActive]      = ImVec4(accDim.x, accDim.y, accDim.z, 0.75f);
    c[ImGuiCol_Button]            = bg1;
    c[ImGuiCol_ButtonHovered]     = bg2;
    c[ImGuiCol_ButtonActive]      = accDim;
    c[ImGuiCol_SliderGrab]        = acc;
    c[ImGuiCol_SliderGrabActive]  = ImVec4(0.40f, 0.66f, 1.0f, 1.0f);
    c[ImGuiCol_CheckMark]         = acc;
    c[ImGuiCol_Tab]               = bg1;
    c[ImGuiCol_TabHovered]        = accDim;
    c[ImGuiCol_SeparatorHovered]  = accDim;
    c[ImGuiCol_SeparatorActive]   = acc;
    c[ImGuiCol_ResizeGrip]        = ImVec4(acc.x, acc.y, acc.z, 0.15f);
    c[ImGuiCol_ResizeGripHovered] = ImVec4(acc.x, acc.y, acc.z, 0.50f);
    c[ImGuiCol_ResizeGripActive]  = acc;
    c[ImGuiCol_DockingPreview]    = ImVec4(acc.x, acc.y, acc.z, 0.40f);
    c[ImGuiCol_NavHighlight]      = acc;
    c[ImGuiCol_Text]              = ImVec4(0.88f, 0.89f, 0.91f, 1.0f);
    c[ImGuiCol_TextDisabled]      = ImVec4(0.45f, 0.47f, 0.51f, 1.0f);
    c[ImGuiCol_PlotLines]         = acc;
    c[ImGuiCol_PlotHistogram]     = acc;
}

// Semantic colors reused across panels.
const ImVec4 kColWarn(0.95f, 0.70f, 0.25f, 1.0f);  // amber: caution
const ImVec4 kColBad(0.93f, 0.35f, 0.32f, 1.0f);   // red: error/stall
const ImVec4 kColGood(0.35f, 0.80f, 0.46f, 1.0f);  // green: in-band/OK
const ImVec4 kColAccent(0.26f, 0.55f, 0.96f, 1.0f);

// ===========================================================================
// Smoothed display values: cheap "animated numbers". Readouts jump every sim
// frame; an exponential approach (frame-rate independent via exp decay) makes
// them glide instead. Keyed by ImGuiID so call sites stay one-liners.
// ===========================================================================

/// @brief Exponentially smooth @p target under the given key; ~90% of a step
/// change is absorbed in 1/speed seconds. Snaps when far off (panel just
/// appeared) so numbers never visibly "count up" from stale state.
float smoothValue(ImGuiID key, float target, float speed = 9.0f) {
    static std::unordered_map<ImGuiID, float> store;
    auto [it, inserted] = store.try_emplace(key, target);
    if (inserted || !std::isfinite(it->second)) {
        it->second = target;
        return target;
    }
    const float dt = ImGui::GetIO().DeltaTime;
    const float a  = 1.0f - std::exp(-speed * dt);
    it->second += (target - it->second) * a;
    // Snap once the residual is visually irrelevant — keeps printf output
    // from dithering in the last digit forever.
    if (std::fabs(it->second - target) < 1e-6f * std::max(1.0f, std::fabs(target)))
        it->second = target;
    return it->second;
}

/// @brief Tooltip helper: dimmed "(?)" marker with hover text, the standard
/// pattern for explaining a readout without consuming panel width.
void helpMarker(const char* text) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

/// @brief Case-insensitive substring match for the airfoil filter box.
bool matchesFilter(const std::string& haystack, const std::string& needle) {
    if (needle.empty()) return true;
    auto lower = [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); };
    std::string h(haystack), n(needle);
    std::transform(h.begin(), h.end(), h.begin(), lower);
    std::transform(n.begin(), n.end(), n.begin(), lower);
    return h.find(n) != std::string::npos;
}

/// @brief Case-insensitive lexicographic less-than, used to sort the Aircraft
/// list by manufacturer regardless of how the CSV capitalizes names.
bool lessCaseInsensitive(const std::string& a, const std::string& b) {
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        const int ca = std::tolower(static_cast<unsigned char>(a[i]));
        const int cb = std::tolower(static_cast<unsigned char>(b[i]));
        if (ca != cb) return ca < cb;
    }
    return a.size() < b.size();
}

// ===========================================================================
// Default docking layout (plan 9.2): edit panels on the left, status panels
// on the right, the 3D scene visible through the passthrough central node.
// Built once when no saved layout exists (imgui.ini wins afterwards).
// ===========================================================================

/// @brief Build the first-run dock layout with the DockBuilder API.
void buildDefaultDockLayout(ImGuiID dockspaceId) {
    ImGui::DockBuilderRemoveNode(dockspaceId);
    ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace
                                          | ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::DockBuilderSetNodeSize(dockspaceId,
                                  ImGui::GetMainViewport()->WorkSize);

    // Left column: geometry editing. Right column: sim health + readouts.
    ImGuiID center = dockspaceId;
    ImGuiID left  = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.22f,
                                                nullptr, &center);
    ImGuiID right = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.26f,
                                                nullptr, &center);
    ImGuiID leftBottom = ImGui::DockBuilderSplitNode(left, ImGuiDir_Down, 0.55f,
                                                     nullptr, &left);
    ImGuiID rightBottom = ImGui::DockBuilderSplitNode(right, ImGuiDir_Down, 0.45f,
                                                      nullptr, &right);

    ImGui::DockBuilderDockWindow("Airfoil", left);
    ImGui::DockBuilderDockWindow("VG Editor", leftBottom);
    ImGui::DockBuilderDockWindow("VG Guidance", leftBottom); // tabbed with editor
    ImGui::DockBuilderDockWindow("Simulation", right);
    ImGui::DockBuilderDockWindow("View", right);             // tabbed with sim
    ImGui::DockBuilderDockWindow("Mesh", right);             // tabbed with sim/view
    ImGui::DockBuilderDockWindow("Readouts", rightBottom);
    ImGui::DockBuilderFinish(dockspaceId);
}

// Central (render passthrough) node rect, refreshed every frame by
// beginDockspace. Overlay widgets (speed legend, etc.) anchor against this
// instead of the full viewport so they never sit on top of docked panels.
ImVec2 gRenderAreaPos(0.0f, 0.0f);
ImVec2 gRenderAreaSize(0.0f, 0.0f);

/// @brief Fullscreen invisible host window + dockspace with a passthrough
/// central node so the GL scene stays visible behind the panels.
void beginDockspace() {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::SetNextWindowViewport(vp->ID);
    ImGuiWindowFlags hostFlags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
        ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoDocking;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("##FoilCFDDockHost", nullptr, hostFlags);
    ImGui::PopStyleVar(2);

    const ImGuiID dockspaceId = ImGui::GetID("FoilCFDDockSpace");
    // First run (no imgui.ini entry for this node): lay out the defaults.
    if (ImGui::DockBuilderGetNode(dockspaceId) == nullptr) {
        buildDefaultDockLayout(dockspaceId);
    }
    ImGui::DockSpace(dockspaceId, ImVec2(0, 0),
                     ImGuiDockNodeFlags_PassthruCentralNode);
    // Remember where the 3D scene actually shows through this frame (the
    // passthrough central node), for edge-anchored overlays.
    if (const ImGuiDockNode* central =
            ImGui::DockBuilderGetCentralNode(dockspaceId)) {
        gRenderAreaPos = central->Pos;
        gRenderAreaSize = central->Size;
    } else {
        gRenderAreaPos = vp->WorkPos;
        gRenderAreaSize = vp->WorkSize;
    }
    ImGui::End();
}

// ===========================================================================
// Global hotkeys (plan 9.1: 1/2/3 view modes; space = run/pause). Suppressed
// while any text field owns the keyboard.
// ===========================================================================

void handleHotkeys(UIContext& ctx) {
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput) return; // typing a NACA code, not toggling views
    UIParams& p = *ctx.params;
    // Mode hotkeys: 1 = particles, 2 = slices, 3 = Q-criterion raycast,
    // 4 = foil mesh, 5 = velocity volume (the default hero "wind-tunnel smoke"
    // look). Modes overlay freely.
    if (ImGui::IsKeyPressed(ImGuiKey_1, false)) p.viz.showParticles = !p.viz.showParticles;
    if (ImGui::IsKeyPressed(ImGuiKey_2, false)) p.viz.showSlices    = !p.viz.showSlices;
    if (ImGui::IsKeyPressed(ImGuiKey_3, false)) p.viz.showQRaycast  = !p.viz.showQRaycast;
    if (ImGui::IsKeyPressed(ImGuiKey_4, false)) p.viz.showFoilMesh  = !p.viz.showFoilMesh;
    if (ImGui::IsKeyPressed(ImGuiKey_5, false)) p.viz.showVelocityVolume = !p.viz.showVelocityVolume;
    if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) p.running = !p.running;
    if (ImGui::IsKeyPressed(ImGuiKey_F, false)) ctx.events->frameFoilView = true;
    // R key: cold-restart the sim (zero field, re-run viscosity ramp).
    if (ImGui::IsKeyPressed(ImGuiKey_R, false)) ctx.events->resetCold = true;
}

// ===========================================================================
// Aircraft section (plan 15.5): a searchable manufacturer/model list mapping
// popular aircraft to their wing sections. Selecting a row with on-disk
// coordinates loads it through the EXACT same path as clicking the .dat list
// (selectedDatIndex + reloadAirfoil) — loading logic lives in one place.
// ===========================================================================

/// @brief Commit one scanned-catalog entry exactly like clicking it in the
/// .dat list box: same params, same event, same main.cpp handler.
void selectCatalogEntry(UIParams& p, UIEvents& ev, int catalogIndex) {
    p.selectedDatIndex = catalogIndex;
    p.source = AirfoilSource::DatFile;
    ev.reloadAirfoil = true;
}

void drawAircraftSection(UIContext& ctx) {
    UIParams& p = *ctx.params;
    UIEvents& ev = *ctx.events;

    const bool open = ImGui::CollapsingHeader("Aircraft");
    // Standing disclaimer ON THE HEADER (plan 15.5: same trust-deltas spirit
    // as the Mission statement) — visible whether or not the list is open.
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
        ImGui::TextUnformatted("Manufacturers modify sections - verify "
                               "against your actual wing before trusting.");
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
    if (!open) return;

    if (!ctx.aircraftManifest || ctx.aircraftManifest->empty()) {
        // Missing/empty manifest degrades to a pointer at the docs, never an
        // error: the CSV is an optional, user-editable data file.
        ImGui::TextDisabled("no aircraft manifest loaded —\n"
                            "see airfoils/MANIFEST_README.md");
        return;
    }
    const std::vector<AircraftEntry>& fleet = *ctx.aircraftManifest;

    // ---- search box: case-insensitive substring over manufacturer+model ----
    {
        char abuf[64] = {};
        std::snprintf(abuf, sizeof(abuf), "%s", p.aircraftFilter.c_str());
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputTextWithHint("##aircraftfilter", "search aircraft...",
                                     abuf, sizeof(abuf))) {
            p.aircraftFilter = abuf;
        }
    }

    // Filter then stable-sort by manufacturer so the list reads as grouped
    // blocks per maker (CSV row order preserved inside each block). 75 rows:
    // rebuilding this every frame costs microseconds.
    std::vector<int> rows;
    rows.reserve(fleet.size());
    for (int i = 0; i < static_cast<int>(fleet.size()); ++i) {
        const AircraftEntry& e = fleet[static_cast<size_t>(i)];
        if (matchesFilter(e.manufacturer + " " + e.model, p.aircraftFilter)) {
            rows.push_back(i);
        }
    }
    std::stable_sort(rows.begin(), rows.end(), [&fleet](int a, int b) {
        return lessCaseInsensitive(fleet[static_cast<size_t>(a)].manufacturer,
                                   fleet[static_cast<size_t>(b)].manufacturer);
    });
    ImGui::TextDisabled("%d / %d aircraft", static_cast<int>(rows.size()),
                        static_cast<int>(fleet.size()));

    ImGui::BeginChild("##aircraftlist",
                      ImVec2(-1, 11.0f * ImGui::GetTextLineHeightWithSpacing()),
                      ImGuiChildFlags_Borders);
    const std::string* prevMaker = nullptr;
    for (const int idx : rows) {
        const AircraftEntry& e = fleet[static_cast<size_t>(idx)];
        ImGui::PushID(idx);

        // Manufacturer group header whenever the maker changes (the list is
        // sorted, so each manufacturer appears exactly once).
        if (!prevMaker || lessCaseInsensitive(*prevMaker, e.manufacturer)
                       || lessCaseInsensitive(e.manufacturer, *prevMaker)) {
            ImGui::TextColored(kColAccent, "%s", e.manufacturer.c_str());
            prevMaker = &e.manufacturer;
        }

        // A row is loadable only when a resolved section is linked into the
        // scanned catalog (the load path REQUIRES a catalog index). When root
        // and tip link to DIFFERENT files, clicking arms the selector below
        // instead of loading immediately (plan 15.5 root/tip toggle).
        const bool loadable = e.catalogIndexRoot >= 0 || e.catalogIndexTip >= 0;
        const bool needsChoice = e.catalogIndexRoot >= 0
                              && e.catalogIndexTip >= 0
                              && e.catalogIndexRoot != e.catalogIndexTip;
        if (ImGui::Selectable(e.model.c_str(), p.selectedAircraftIndex == idx,
                              loadable ? 0 : ImGuiSelectableFlags_Disabled)) {
            p.selectedAircraftIndex = idx;
            if (!needsChoice) {
                // One section on disk (or root == tip): load it right away,
                // preferring the root per the plan's "loads dat_root".
                selectCatalogEntry(p, ev, e.catalogIndexRoot >= 0
                                              ? e.catalogIndexRoot
                                              : e.catalogIndexTip);
            }
        }
        // Hover state captured NOW — the badge drawn next is its own item.
        const bool hovered =
            ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled);

        // Category badge, right-aligned dim text on the selectable's row.
        if (!e.category.empty()) {
            const float w = ImGui::CalcTextSize(e.category.c_str()).x;
            ImGui::SameLine(ImGui::GetContentRegionMax().x - w
                            - ImGui::GetStyle().FramePadding.x);
            ImGui::TextDisabled("%s", e.category.c_str());
        }

        // Row tooltip: the documented sections plus the notes column.
        if (hovered) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
            ImGui::Text("Root: %s", e.airfoilRoot.empty()
                                        ? "(unknown)" : e.airfoilRoot.c_str());
            if (!e.airfoilTip.empty() && e.airfoilTip != e.airfoilRoot) {
                ImGui::Text("Tip:  %s", e.airfoilTip.c_str());
            }
            if (!e.notes.empty()) ImGui::TextDisabled("%s", e.notes.c_str());
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }

        if (!loadable) {
            // Greyed row: surface WHY inline (the notes column), plus the
            // nearest-section hint for the NACA 23013/23014 Van's RV fleet
            // (23012 and 23015 bracket those half-designations on disk).
            ImGui::Indent();
            ImGui::PushStyleColor(ImGuiCol_Text,
                                  ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            ImGui::TextWrapped("%s", e.notes.empty()
                                         ? "coordinates not in database"
                                         : e.notes.c_str());
            ImGui::PopStyleColor();
            if (e.airfoilRoot.find("23013") != std::string::npos
                || e.airfoilRoot.find("23014") != std::string::npos) {
                ImGui::TextColored(kColAccent,
                                   "closest in database: naca23012 / naca23015");
            }
            ImGui::Unindent();
        } else if (p.selectedAircraftIndex == idx && needsChoice) {
            // Root/tip selector: distinct sections at root and tip — the
            // user picks WHICH station to simulate before anything loads.
            ImGui::Indent();
            ImGui::TextDisabled("root and tip differ — load which section?");
            char lbl[96];
            std::snprintf(lbl, sizeof(lbl), "Root (%s)", e.datRoot.c_str());
            if (ImGui::SmallButton(lbl)) {
                selectCatalogEntry(p, ev, e.catalogIndexRoot);
            }
            ImGui::SameLine();
            std::snprintf(lbl, sizeof(lbl), "Tip (%s)", e.datTip.c_str());
            if (ImGui::SmallButton(lbl)) {
                selectCatalogEntry(p, ev, e.catalogIndexTip);
            }
            ImGui::Unindent();
        }
        ImGui::PopID();
    }
    ImGui::EndChild();
}

// ===========================================================================
// Panel: Airfoil (plan 9.2 first bullet + High Fidelity from the Mission
// statement). Source selection, the FILTERED catalog list (the UIUC database
// is ~1600 files — a flat dropdown is unusable), NACA box, physical inputs,
// AoA on slider RELEASE, and the resolution presets.
// ===========================================================================

void drawAirfoilPanel(UIContext& ctx) {
    UIParams& p = *ctx.params;
    UIReadouts& r = *ctx.readouts;
    UIEvents& ev = *ctx.events;
    if (!ImGui::Begin("Airfoil")) { ImGui::End(); return; }

    ImGui::TextDisabled("GEOMETRY");
    ImGui::Text("Active: %s", r.loadedAirfoilName.empty()
                                  ? "(none)" : r.loadedAirfoilName.c_str());
    ImGui::Spacing();

    // ---- NACA 4-digit generator ----
    {
        char buf[16] = {};
        std::snprintf(buf, sizeof(buf), "%s", p.nacaDigits.c_str());
        ImGui::SetNextItemWidth(90.0f);
        // Enter-to-commit so half-typed codes don't trigger regeneration.
        if (ImGui::InputText("NACA", buf, sizeof(buf),
                             ImGuiInputTextFlags_CharsDecimal |
                             ImGuiInputTextFlags_EnterReturnsTrue)) {
            p.nacaDigits = buf;
            p.source = AirfoilSource::NacaDigits;
            ev.reloadAirfoil = true;
        } else {
            p.nacaDigits = buf; // keep edits without committing
        }
        ImGui::SameLine();
        if (ImGui::Button("Generate")) {
            p.source = AirfoilSource::NacaDigits;
            ev.reloadAirfoil = true;
        }
    }

    // ---- UIUC .dat catalog: filter box + clipped list ----
    ImGui::Spacing();
    {
        char fbuf[64] = {};
        std::snprintf(fbuf, sizeof(fbuf), "%s", p.airfoilFilter.c_str());
        ImGui::SetNextItemWidth(-70.0f);
        if (ImGui::InputTextWithHint("##foilfilter", "filter airfoils...",
                                     fbuf, sizeof(fbuf))) {
            p.airfoilFilter = fbuf;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Rescan")) ev.refreshAirfoils = true;

        // Build the filtered index list each frame: 1600 substring tests is
        // microseconds; the clipper keeps the draw cost at ~20 rows.
        std::vector<int> filtered;
        if (ctx.airfoilCatalog) {
            filtered.reserve(ctx.airfoilCatalog->size());
            for (int i = 0; i < static_cast<int>(ctx.airfoilCatalog->size()); ++i) {
                if (matchesFilter((*ctx.airfoilCatalog)[i].displayName,
                                  p.airfoilFilter)) {
                    filtered.push_back(i);
                }
            }
        }
        ImGui::TextDisabled("%d / %d files", static_cast<int>(filtered.size()),
                            ctx.airfoilCatalog
                                ? static_cast<int>(ctx.airfoilCatalog->size()) : 0);
        if (ImGui::BeginListBox("##foillist", ImVec2(-1, 8.5f * ImGui::GetTextLineHeightWithSpacing()))) {
            ImGuiListClipper clipper;
            clipper.Begin(static_cast<int>(filtered.size()));
            while (clipper.Step()) {
                for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                    const int idx = filtered[static_cast<size_t>(row)];
                    const bool selected = (p.selectedDatIndex == idx
                                           && p.source == AirfoilSource::DatFile);
                    ImGui::PushID(idx);
                    if (ImGui::Selectable((*ctx.airfoilCatalog)[idx].displayName.c_str(),
                                          selected)) {
                        p.selectedDatIndex = idx;
                        p.source = AirfoilSource::DatFile;
                        ev.reloadAirfoil = true;
                    }
                    ImGui::PopID();
                }
            }
            ImGui::EndListBox();
        }
    }

    // ---- Aircraft search (plan 15.5): pick the plane, get its airfoil ----
    ImGui::Spacing();
    drawAircraftSection(ctx);
    ImGui::Spacing();

    if (ImGui::Button("Load STL...", ImVec2(-1, 0))) ev.loadStlRequested = true;
    helpMarker("Import a watertight solid (binary or ASCII STL, up to 2M "
               "triangles). You can also drag-and-drop .stl or .dat files "
               "onto the window.");

    // ---- flow conditions ----
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("FLOW");
    const bool stlMode = (p.source == AirfoilSource::StlImport);
    ImGui::BeginDisabled(stlMode);
    // AoA: re-voxelize ONLY on release (plan 13) — dragging is free.
    ImGui::SliderFloat("AoA", &p.aoaDeg, -5.0f, 20.0f, "%.1f deg");
    if (ImGui::IsItemDeactivatedAfterEdit()) ev.aoaChanged = true;
    ImGui::EndDisabled();
    if (stlMode) {
        ImGui::TextDisabled("AoA applies to airfoil sections only —\n"
                            "orient STL solids in CAD before export.");
    }

    // Airspeed: stored canonically in m/s, but shown/edited in the user's
    // chosen unit (knots by default — the aviation convention). We convert the
    // m/s value and the 5..120 m/s range into the display unit, let the slider
    // edit in that unit, then convert the result straight back to m/s so the
    // physics path never sees anything but m/s.
    {
        const float k = speedUnitPerMs(p.speedUnit);
        float shown   = p.airspeedMs * k;
        char fmt[24];
        std::snprintf(fmt, sizeof fmt, "%%.1f %s", speedUnitLabel(p.speedUnit));
        if (ImGui::SliderFloat("Airspeed", &shown, 5.0f * k, 120.0f * k, fmt)) {
            p.airspeedMs = shown / k; // commit back in canonical m/s
        }
        if (ImGui::IsItemDeactivatedAfterEdit()) ev.airspeedChanged = true;

        // Unit toggle buttons: knots / mph / m/s. Selected unit is highlighted;
        // switching only changes the display (airspeedMs is unchanged, so no
        // re-sim event fires).
        auto unitButton = [&](const char* label, SpeedUnit u) {
            const bool active = (p.speedUnit == u);
            if (active)
                ImGui::PushStyleColor(ImGuiCol_Button,
                                      ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
            if (ImGui::SmallButton(label)) p.speedUnit = u;
            if (active) ImGui::PopStyleColor();
        };
        unitButton("kn", SpeedUnit::Knots);
        ImGui::SameLine();
        unitButton("mph", SpeedUnit::Mph);
        ImGui::SameLine();
        unitButton("m/s", SpeedUnit::Ms);
    }
    helpMarker("Changing airspeed rescales the units and restarts the "
               "simulation. The kn/mph/m-s buttons only change the display "
               "unit — the simulation always runs in SI.");

    // Chord is a units rescale exactly like airspeed: grid dims, u_lat, and
    // the flag field are untouched (only dx/dt/Re-target change); the flow
    // restarts cleanly on release.
    ImGui::SliderFloat("Chord", &p.chordM, 0.2f, 4.0f, "%.2f m");
    if (ImGui::IsItemDeactivatedAfterEdit()) ev.airspeedChanged = true;
    helpMarker("Physical chord length. Like airspeed, this rescales the unit "
               "conversion (target Reynolds number) and restarts the "
               "simulation.");

    // ---- resolution presets + High Fidelity ----
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("RESOLUTION");
    ImGui::BeginDisabled(p.highFidelity.enabled);
    {
        // Fast/Default/Fine only — Ultra is reserved for High Fidelity mode.
        static const char* kPresetNames[] = {"Fast (192 cells/chord)",
                                             "Default (256 cells/chord)",
                                             "Fine (320 cells/chord)"};
        int presetIdx = std::min(static_cast<int>(p.resolution), 2);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##respreset", &presetIdx, kPresetNames, 3)) {
            p.resolution = static_cast<ResolutionPreset>(presetIdx);
            ev.resolutionChanged = true;
        }
    }
    ImGui::EndDisabled();

    bool hifi = p.highFidelity.enabled;
    if (ImGui::Checkbox("High Fidelity mode", &hifi)) {
        p.highFidelity.enabled = hifi;
        // The Mission-statement bundle: Ultra chord resolution, u_lat 0.05
        // (units.h HighFidelityPreset default), long force EMA window.
        p.highFidelity.resolution = ResolutionPreset::Ultra;
        ev.highFidelityToggled = true;
    }
    helpMarker("Trades interactivity for accuracy: Ultra grid (384 cells/"
               "chord), lattice speed lowered to 0.05 (quarter the "
               "compressibility error), and force averaging over 8 flow-"
               "throughs. Use it for the final VG-on vs VG-off comparison.");

    // Show what the active preset means in cells and memory so the user can
    // anticipate the VRAM bill before committing (plan 4.6 sanity numbers).
    {
        const int nc = p.highFidelity.enabled
                           ? chordCellsFor(p.highFidelity.resolution)
                           : chordCellsFor(p.resolution);
        const long long nx = 3LL * nc, ny = (5LL * nc) / 4, nz = (3LL * nc) / 8;
        const double cells = static_cast<double>(nx * ny * nz);
        // 152 B f-pair + 16 B macroscopic + 1 B flags + clean snapshot share.
        const double gb = cells * (152.0 + 16.0 + 1.0) / (1024.0 * 1024.0 * 1024.0);
        ImGui::TextDisabled("grid %lld x %lld x %lld  (~%.1f GB VRAM)",
                            nx, ny, nz, gb);
    }
    ImGui::End();
}

// ===========================================================================
// Panel: VG editor (plan 9.2 second bullet, params per section 6.1). Every
// continuous param commits on slider RELEASE -> vgEdited -> warm restart;
// discrete edits (type, count, add/dup/delete) commit immediately.
// ===========================================================================

/// @brief Draw the parameter widgets for one VG entry; returns true when a
/// sim-affecting edit was committed this frame.
/// @param xcMin  Lower x/c bound for the Station slider (user-configured range).
/// @param xcMax  Upper x/c bound for the Station slider.
bool drawVGEntry(VGParams& vg, int chordCells, float xcMin, float xcMax) {
    bool edited = false;

    // Type combo — discrete, commits immediately.
    static const char* kTypeNames[] = {"Single vane", "Counter-rotating pair",
                                       "Co-rotating array", "Ramp"};
    int typeIdx = static_cast<int>(vg.type);
    if (ImGui::Combo("Type", &typeIdx, kTypeNames, 4)) {
        vg.type = static_cast<VGType>(typeIdx);
        edited = true;
    }

    // Helper: slider that fires on release, followed by a tooltip marker.
    auto releaseSlider = [&edited](const char* label, float* v, float lo,
                                   float hi, const char* fmt,
                                   const char* tooltip) {
        ImGui::SliderFloat(label, v, lo, hi, fmt);
        if (ImGui::IsItemDeactivatedAfterEdit()) edited = true;
        ImGui::SameLine();
        helpMarker(tooltip);
    };

    // Station x/c slider uses the user-configured chord range.
    // Clamp the current value into the allowed range first so it can't
    // sit outside the window when the range is tightened.
    vg.x_c = std::clamp(vg.x_c, xcMin, xcMax);
    releaseSlider("Station x/c", &vg.x_c, xcMin, xcMax, "%.3f",
                  "Chordwise position of the VG leading edge as a fraction of "
                  "chord (0 = leading edge, 1 = trailing edge). Lin (2002) "
                  "recommends placing VGs 5–10 device heights upstream of the "
                  "separation onset shown in the Guidance panel. "
                  "The allowed range is set by the Min/Max controls above.");

    releaseSlider("Height h/c", &vg.height_c, 0.002f, 0.030f, "%.4f",
                  "VG device height as a fraction of chord. "
                  "h sets the length of the streamwise vortex — taller vanes "
                  "energise deeper into the boundary layer but add more drag. "
                  "Lin (2002) nominal: h ~ delta99 at the placement station "
                  "(shown in the Guidance panel). Typical range: 0.005–0.020 c.");

    releaseSlider("Length (h)", &vg.length_h, 1.0f, 6.0f, "%.1f",
                  "Vane chord length expressed as multiples of device height h. "
                  "Longer vanes generate stronger vortices but increase drag. "
                  "Strausak flight-proven value: 3 h. Typical range: 2–4 h.");

    releaseSlider("Incidence", &vg.beta_deg, -30.0f, 30.0f, "%.1f deg",
                  "Vane incidence angle relative to the freestream [degrees]. "
                  "Positive = swept toward +z (right). Counter-rotating pairs "
                  "use +/- symmetric angles. Strausak: ~16 deg. "
                  "Higher angles produce stronger vortices with more drag penalty.");

    const bool multiUnit = (vg.type == VGType::CounterRotatingPair
                            || vg.type == VGType::CoRotatingArray);
    if (multiUnit) {
        releaseSlider("Pitch (c)", &vg.pitch_c, 0.01f, 0.20f, "%.3f",
                      "Spanwise spacing between adjacent VG units as a fraction "
                      "of chord. Tighter pitches give more uniform spanwise "
                      "re-energisation; too tight and vortices merge. "
                      "Typical: 3–6 h (expressed here in chord units).");
    }
    if (vg.type == VGType::CounterRotatingPair) {
        releaseSlider("Gap (h)", &vg.gap_h, 1.0f, 6.0f, "%.1f",
                      "Lateral gap between the two vanes in a counter-rotating "
                      "pair, in multiples of h. Controls the proximity of the "
                      "two vortex cores. Strausak: ~2 h.");
        bool cfd = vg.commonFlowDown;
        if (ImGui::Checkbox("Common-flow-down", &cfd)) {
            vg.commonFlowDown = cfd;
            edited = true;
        }
        ImGui::SameLine();
        helpMarker("When checked the two vanes are angled so their common-flow "
                   "region points DOWN toward the surface (common-flow-down / "
                   "inboard-pointing arrangement). This is the typical high-lift "
                   "configuration. Uncheck for common-flow-up.");
    }
    if (multiUnit) {
        int count = vg.count;
        if (ImGui::SliderInt("Units", &count, 1, 16)) { /* drag preview */ }
        if (ImGui::IsItemDeactivatedAfterEdit()) edited = true;
        vg.count = count;
        ImGui::SameLine();
        helpMarker("Number of VG units in the spanwise array. Each unit is one "
                   "full device (or pair) separated by Pitch from its neighbour. "
                   "The sim runs all units simultaneously.");
    }

    // Under-resolution guard (plan 6.1): warn instead of rendering noise.
    if (vgUnderResolved(vg, chordCells)) {
        ImGui::TextColored(kColWarn,
                           "VG under-resolved (%.1f cells tall, need %d) —\n"
                           "increase chord resolution or VG size",
                           vgHeightCells(vg, chordCells), kMinVGHeightCells);
    } else {
        ImGui::TextDisabled("vane height: %.1f cells",
                            vgHeightCells(vg, chordCells));
    }
    return edited;
}

void drawVGEditorPanel(UIContext& ctx) {
    UIParams& p = *ctx.params;
    UIEvents& ev = *ctx.events;
    // Under-resolution checks judge against the grid the vanes are actually
    // voxelized on: the 2x fine patch when it is active (plan M-refine —
    // doubling vane resolution is the patch's headline win), else the base.
    const int chordCells = (ctx.readouts->refine.active ? 2 : 1)
                         * ctx.readouts->scaling.chordCells;
    if (!ImGui::Begin("VG Editor")) { ImGui::End(); return; }

    if (p.source == AirfoilSource::StlImport) {
        // Plan 7.4: no parametric surface on an STL, so the surface-frame
        // editor is unavailable (the free-placement gizmo is a v1.x feature).
        ImGui::TextWrapped("VG surface placement needs a parametric airfoil "
                           "section. Load a NACA or .dat foil to edit VGs; "
                           "free 3D placement for STL solids is planned.");
        ImGui::End();
        return;
    }

    ImGui::TextDisabled("Each edit restarts the sim from cold.");

    // --- Station x/c placement range controls --------------------------------
    // Let the user restrict which chord region VGs can be placed in.  Editing
    // the range immediately clamps any out-of-bounds VG stations and restarts.
    ImGui::Separator();
    ImGui::TextDisabled("Placement range (x/c %)");
    helpMarker("Restricts the Station slider to a chord-percentage window.  "
               "Useful when you know separation occurs in a specific band and "
               "want to prevent accidental placement outside it.");
    float xcMinPct = p.vgXcMin * 100.0f;
    float xcMaxPct = p.vgXcMax * 100.0f;
    ImGui::SetNextItemWidth(80.0f);
    if (ImGui::InputFloat("Min %", &xcMinPct, 1.0f, 5.0f, "%.0f")) {
        // Clamp and enforce min < max - 1%.
        xcMinPct = std::clamp(xcMinPct, 0.5f, xcMaxPct - 1.0f);
        p.vgXcMin = xcMinPct / 100.0f;
        // Clamp any existing VG stations into the new range.
        bool anyOutside = false;
        for (auto& vg : p.vgs) {
            const float prev = vg.x_c;
            vg.x_c = std::clamp(vg.x_c, p.vgXcMin, p.vgXcMax);
            if (vg.x_c != prev) anyOutside = true;
        }
        if (anyOutside) ev.vgEdited = true;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.0f);
    if (ImGui::InputFloat("Max %", &xcMaxPct, 1.0f, 5.0f, "%.0f")) {
        xcMaxPct = std::clamp(xcMaxPct, xcMinPct + 1.0f, 70.0f);
        p.vgXcMax = xcMaxPct / 100.0f;
        bool anyOutside = false;
        for (auto& vg : p.vgs) {
            const float prev = vg.x_c;
            vg.x_c = std::clamp(vg.x_c, p.vgXcMin, p.vgXcMax);
            if (vg.x_c != prev) anyOutside = true;
        }
        if (anyOutside) ev.vgEdited = true;
    }
    ImGui::Separator();

    if (ImGui::Button("+ Add VG")) {
        p.vgs.push_back(defaultVGParams());
        p.selectedVG = static_cast<int>(p.vgs.size()) - 1;
        ev.vgEdited = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Strausak preset")) {
        // Flight-proven recipe (Mission statement): x/c 0.07, +/-15-16 deg
        // counter-rotating pairs, l = 3h — exactly defaultVGParams().
        p.vgs.clear();
        p.vgs.push_back(defaultVGParams());
        p.selectedVG = 0;
        ev.vgEdited = true;
    }
    helpMarker("Flight-proven preset (Strausak): counter-rotating vane pairs "
               "at x/c = 0.07, ~16 deg incidence, length 3h. A solid starting "
               "point before tuning against the guidance panel.");

    int deleteIdx = -1, duplicateIdx = -1;
    for (int i = 0; i < static_cast<int>(p.vgs.size()); ++i) {
        ImGui::PushID(i);
        char header[64];
        std::snprintf(header, sizeof(header), "VG %d  (x/c %.2f)###vg%d",
                      i + 1, p.vgs[static_cast<size_t>(i)].x_c, i);
        const bool open = ImGui::CollapsingHeader(
            header, i == p.selectedVG ? ImGuiTreeNodeFlags_DefaultOpen : 0);
        if (ImGui::IsItemClicked()) p.selectedVG = i; // guidance panel anchor
        if (open) {
            ImGui::Indent();
            if (drawVGEntry(p.vgs[static_cast<size_t>(i)], chordCells, p.vgXcMin, p.vgXcMax)) {
                p.selectedVG = i;
                ev.vgEdited = true;
            }
            if (ImGui::SmallButton("Duplicate")) duplicateIdx = i;
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, kColBad);
            if (ImGui::SmallButton("Delete")) deleteIdx = i;
            ImGui::PopStyleColor();
            ImGui::Unindent();
            ImGui::Spacing();
        }
        ImGui::PopID();
    }

    // List mutations after the loop so indices stay coherent mid-iteration.
    if (duplicateIdx >= 0) {
        p.vgs.insert(p.vgs.begin() + duplicateIdx + 1,
                     p.vgs[static_cast<size_t>(duplicateIdx)]);
        p.selectedVG = duplicateIdx + 1;
        ev.vgEdited = true;
    }
    if (deleteIdx >= 0) {
        p.vgs.erase(p.vgs.begin() + deleteIdx);
        p.selectedVG = std::min(p.selectedVG,
                                static_cast<int>(p.vgs.size()) - 1);
        ev.vgEdited = true;
    }
    if (p.vgs.empty()) {
        ImGui::TextDisabled("No VGs — clean foil. Add one to compare deltas.");
    }
    ImGui::End();
}

// ===========================================================================
// Panel: VG Guidance (Mission statement). Sim-derived delta99 at the selected
// station, the Lin-2002 h band, the station band 5-10h upstream of separation,
// and the under-resolution warning — the numbers a homebuilder acts on.
// ===========================================================================

void drawVGGuidancePanel(UIContext& ctx) {
    UIParams& p = *ctx.params;
    UIReadouts& r = *ctx.readouts;
    UIEvents& ev = *ctx.events;
    if (!ImGui::Begin("VG Guidance")) { ImGui::End(); return; }

    if (p.source == AirfoilSource::StlImport) {
        ImGui::TextWrapped("Boundary-layer guidance needs the parametric "
                           "suction surface of an airfoil section.");
        ImGui::End();
        return;
    }

    ImGui::Checkbox("Show guidance overlay", &p.showVGGuidanceOverlay);
    ImGui::Separator();

    // The station/height the guidance anchors to: the selected VG entry, or
    // the Strausak station as a sensible default before any VG exists.
    const bool haveVG = p.selectedVG >= 0
                     && p.selectedVG < static_cast<int>(p.vgs.size());
    const float stationXc = haveVG
        ? p.vgs[static_cast<size_t>(p.selectedVG)].x_c : 0.07f;
    const float heightC = haveVG
        ? p.vgs[static_cast<size_t>(p.selectedVG)].height_c : 0.01f;
    ImGui::Text("Station x/c = %.3f  %s", stationXc,
                haveVG ? "(selected VG)" : "(default — add a VG to track it)");

    // Gate everything on a minimally developed flow: delta99 from the first
    // flow-through is start-up transient, not boundary layer.
    if (r.flowThroughs < 1.0f) {
        ImGui::TextDisabled("Developing flow... %.1f / 1.0 flow-throughs "
                            "before BL data is meaningful.", r.flowThroughs);
        ImGui::End();
        return;
    }

    // ---- delta99 at the selected station (nearest profile sample) ----
    const Delta99Sample* nearest = nullptr;
    for (const Delta99Sample& s : r.delta99Profile) {
        if (!s.valid) continue;
        if (!nearest || std::fabs(s.x_c - stationXc)
                            < std::fabs(nearest->x_c - stationXc)) {
            nearest = &s;
        }
    }
    if (nearest) {
        const float d99mm = nearest->delta99_c * p.chordM * 1000.0f;
        ImGui::Text("delta99 = %.4f c  (%.1f mm at %.2f m chord)",
                    smoothValue(ImGui::GetID("d99"), nearest->delta99_c),
                    smoothValue(ImGui::GetID("d99mm"), d99mm), p.chordM);
        helpMarker("Boundary-layer thickness measured wall-normal from the "
                   "suction surface in the live simulation, at the sample "
                   "station nearest your VG.");
    } else {
        ImGui::TextColored(kColWarn, "delta99 unavailable at this station\n"
                                     "(separated or off-surface probe).");
    }

    // ---- Lin 2002 height band: h = 0.1..1.0 * delta99 ----
    if (r.heightBand.valid) {
        const float loMm = r.heightBand.minVal * p.chordM * 1000.0f;
        const float hiMm = r.heightBand.maxVal * p.chordM * 1000.0f;
        ImGui::Text("Lin 2002 height band:");
        ImGui::Text("  h/c %.4f .. %.4f  (%.1f .. %.1f mm)",
                    r.heightBand.minVal, r.heightBand.maxVal, loMm, hiMm);
        // Verdict line: is the configured h inside the published band?
        if (heightC < r.heightBand.minVal) {
            ImGui::TextColored(kColWarn, "  current h/c %.4f is BELOW the band", heightC);
        } else if (heightC > r.heightBand.maxVal) {
            ImGui::TextColored(kColWarn, "  current h/c %.4f is ABOVE the band", heightC);
        } else {
            ImGui::TextColored(kColGood, "  current h/c %.4f is in band", heightC);
        }
        helpMarker("Lin (2002): low-profile VGs work best at h = 0.1-1.0 x "
                   "the local boundary-layer thickness. Sub-delta99 devices "
                   "(h ~ 0.2 delta99) give most of the separation control "
                   "with far less device drag.");
    } else {
        ImGui::TextDisabled("Height band: waiting for valid delta99.");
    }

    ImGui::Spacing();
    // ---- separation onset + station band 5-10h upstream (Lin 2002) ----
    if (r.separationXc >= 0.0f) {
        ImGui::Text("Separation onset: x/c = %.3f",
                    smoothValue(ImGui::GetID("sepxc"), r.separationXc));
        if (r.stationBand.valid) {
            ImGui::Text("Recommended station: x/c %.3f .. %.3f",
                        r.stationBand.minVal, r.stationBand.maxVal);
            const bool inBand = stationXc >= r.stationBand.minVal
                             && stationXc <= r.stationBand.maxVal;
            ImGui::TextColored(inBand ? kColGood : kColWarn,
                               inBand ? "  selected station is in band"
                                      : "  selected station is OUT of band");
            helpMarker("Lin (2002): place VGs 5-10 device heights upstream of "
                       "separation onset so the streamwise vortices are fully "
                       "developed where the flow needs the momentum.");
        }
    } else {
        ImGui::TextColored(kColGood, "Flow attached — no separation detected.");
        ImGui::TextDisabled("Raise AoA to find the separation point this\n"
                            "airfoil needs VGs for.");
    }

    // ---- delta99 profile sparkline along the chord ----
    if (!r.delta99Profile.empty()) {
        ImGui::Spacing();
        std::vector<float> plot;
        plot.reserve(r.delta99Profile.size());
        for (const Delta99Sample& s : r.delta99Profile)
            plot.push_back(s.valid ? s.delta99_c : 0.0f);
        ImGui::PlotLines("##d99profile", plot.data(),
                         static_cast<int>(plot.size()), 0,
                         "delta99(x/c) on suction surface", 0.0f, FLT_MAX,
                         ImVec2(-1, 56));
    }

    // Under-resolution cross-check mirrored here because this is the panel
    // the user reads while choosing h (geom/vg guard, plan 6.1).
    if (haveVG && vgUnderResolved(p.vgs[static_cast<size_t>(p.selectedVG)],
                                  r.scaling.chordCells)) {
        ImGui::TextColored(kColWarn, "Selected VG is under-resolved on this "
                                     "grid —\nresults will understate its effect.");
    }

    ImGui::Spacing();
    if (ImGui::Button("Apply Strausak defaults (flight-proven)", ImVec2(-1, 0))) {
        p.vgs.clear();
        p.vgs.push_back(defaultVGParams());
        p.selectedVG = 0;
        ev.vgEdited = true;
    }
    ImGui::End();
}

// ===========================================================================
// Panel: Simulation (plan 9.2 third bullet + plan 4.3 honesty numbers + the
// plan 4.5 NaN watchdog surface).
// ===========================================================================

void drawSimPanel(UIContext& ctx) {
    UIParams& p = *ctx.params;
    UIReadouts& r = *ctx.readouts;
    UIEvents& ev = *ctx.events;
    if (!ImGui::Begin("Simulation")) { ImGui::End(); return; }

    // ---- CUDA failure surface (plan 4.5 TDR robustness): a launch/interop
    // error latched by the frame loop pauses the sim for the session and is
    // reported here instead of silently spinning on a frozen field ----
    if (r.cudaErrorTripped) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.30f, 0.08f, 0.08f, 1.0f));
        ImGui::BeginChild("##cudabox", ImVec2(-1, 0),
                          ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Borders);
        ImGui::TextColored(kColBad, "GPU ERROR — simulation stopped");
        ImGui::TextWrapped("%s", r.cudaErrorDiagnosis.c_str());
        ImGui::TextWrapped("Restart FoilCFD to recover. If this follows a "
                           "display timeout on a huge grid, see the TdrDelay "
                           "note in BUILDING.md.");
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    // ---- NaN watchdog error surface: loud, on top, with the diagnosis ----
    if (r.nanTripped) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.30f, 0.08f, 0.08f, 1.0f));
        ImGui::BeginChild("##nanbox", ImVec2(-1, 0),
                          ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Borders);
        ImGui::TextColored(kColBad, "SIMULATION DIVERGED (NaN detected)");
        ImGui::TextWrapped("%s", r.nanDiagnosis.c_str());
        if (ImGui::Button("Cold reset")) ev.resetCold = true;
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    // ---- transport controls ----
    if (ImGui::Button(p.running ? "Pause  [space]" : "Run  [space]",
                      ImVec2(120, 0))) {
        p.running = !p.running;
    }
    ImGui::SameLine();
    if (ImGui::Button("Step")) ev.singleStep = true;
    ImGui::SameLine();
    if (ImGui::Button("Cold reset")) ev.resetCold = true;


    // ---- the honesty line (plan 4.3: DISPLAY BOTH Re numbers) ----
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("REYNOLDS");
    ImGui::Text("simulating at Re %.2e", r.scaling.reEffective);
    ImGui::Text("target Re %.2e", r.scaling.reTarget);
    if (r.scaling.tauClamped) {
        ImGui::TextColored(kColWarn, "tau clamped: target Re unreachable at\n"
                                     "this resolution. Trends/deltas remain\n"
                                     "meaningful; absolutes carry uncertainty.");
    }
    helpMarker("The lattice cannot reach flight Reynolds numbers at this "
               "resolution; the solver runs at the highest stable effective "
               "Re instead. Vortex topology, separation behavior, and VG-on "
               "vs VG-off deltas remain actionable.");

    // ---- live solver numbers ----
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("SOLVER");
    ImGui::Text("steps/frame: %d", r.perf.lastStepsPerFrame);
    ImGui::Text("MLUPS: %.0f",
                smoothValue(ImGui::GetID("mlups"),
                            static_cast<float>(r.perf.mlups)));
    ImGui::Text("step: %lld   flow-throughs: %.2f", r.stepCount, r.flowThroughs);
    ImGui::Text("tau: %.4f (target %.4f)   u_lat: %.3f",
                r.currentTau, r.scaling.tau, r.scaling.u_lat);
    if (r.currentTau > r.scaling.tau + 1e-5f) {
        ImGui::TextDisabled("startup viscosity ramp active");
    }

    // ---- VRAM budget (plan 4.6: warn above 80%) ----
    ImGui::Spacing();
    char vramLabel[48];
    std::snprintf(vramLabel, sizeof(vramLabel), "VRAM %.0f%%",
                  r.vramUsedFraction * 100.0f);
    if (r.vramUsedFraction > 0.8f) {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, kColBad);
        ImGui::ProgressBar(r.vramUsedFraction, ImVec2(-1, 0), vramLabel);
        ImGui::PopStyleColor();
        ImGui::TextColored(kColWarn, "VRAM > 80%% — consider a smaller preset.");
    } else {
        ImGui::ProgressBar(r.vramUsedFraction, ImVec2(-1, 0), vramLabel);
    }
    ImGui::End();
}

// ===========================================================================
// Panel: Readouts (plan 9.2 fourth bullet). Cl/Cd/L-over-D EMA, gated until
// 2 flow-throughs (plan 13), with the Mission-statement trust-deltas tooltip.
// ===========================================================================

void drawReadoutsPanel(UIContext& ctx) {
    UIReadouts& r = *ctx.readouts;
    if (!ImGui::Begin("Readouts")) { ImGui::End(); return; }

    ImGui::TextDisabled("FORCES (EMA)");
    helpMarker("Trust DELTAS, not absolutes: compare VG-on vs VG-off on the "
               "identical grid and settings — that difference is the product. "
               "Absolute Cl/Cd at flight Reynolds numbers carry uncertainty "
               "at this fidelity. Always verify a VG install with tuft or "
               "cotton-tape testing before drilling holes in your wing.");

    if (!r.forces.valid) {
        // Gated: the first two flow-throughs of any cold start are transient
        // garbage for forces (plan 13) — show progress, not numbers.
        ImGui::TextDisabled("converging...  %.1f / %.1f flow-throughs",
                            r.forces.flowThroughs, kForceGateFlowThroughs);
        const float frac = std::clamp(
            r.forces.flowThroughs / kForceGateFlowThroughs, 0.0f, 1.0f);
        ImGui::ProgressBar(frac, ImVec2(-1, 0), "");
    } else {
        // Smoothed display values so the numbers glide rather than flicker.
        const float cl  = smoothValue(ImGui::GetID("cl"),  r.forces.cl);
        const float cd  = smoothValue(ImGui::GetID("cd"),  r.forces.cd);
        const float ld  = smoothValue(ImGui::GetID("ld"),  r.forces.liftToDrag);
        const float cla = smoothValue(ImGui::GetID("cla"), r.forces.clAvg);
        const float cda = smoothValue(ImGui::GetID("cda"), r.forces.cdAvg);

        // Large-type readout: 1.6x font scale on the three hero numbers.
        ImGui::SetWindowFontScale(1.6f);
        ImGui::Text("Cl  %+.4f", cl);
        ImGui::SetWindowFontScale(1.0f);
        ImGui::SameLine();
        helpMarker("Lift coefficient: dimensionless vertical force normalised by "
                   "0.5 * rho * V^2 * chord * span. Positive = upward lift. "
                   "This is the live EMA — compare VG-on vs VG-off deltas; "
                   "absolute values carry LBM fidelity uncertainty.");

        ImGui::SetWindowFontScale(1.6f);
        ImGui::Text("Cd  %.5f", cd);
        ImGui::SetWindowFontScale(1.0f);
        ImGui::SameLine();
        helpMarker("Drag coefficient: dimensionless streamwise force normalised "
                   "by 0.5 * rho * V^2 * chord * span. Includes pressure drag "
                   "and resolved skin friction at this grid resolution. "
                   "Use deltas (VG-on minus VG-off) for engineering guidance.");

        ImGui::SetWindowFontScale(1.6f);
        ImGui::Text("L/D %.2f", ld);
        ImGui::SetWindowFontScale(1.0f);
        ImGui::SameLine();
        helpMarker("Lift-to-drag ratio: Cl / Cd. Higher is more aerodynamically "
                   "efficient. A VG that raises L/D means it recovered more lift "
                   "than it added drag — the key homebuilder metric.");

        // --- Trailing averages over the last 1.0 flow-through ---
        // These settle faster than the EMA and show the mean over a physically
        // complete cycle, giving a cleaner comparison point across restarts.
        ImGui::Spacing();
        ImGui::TextDisabled("2-flow-through avg");
        helpMarker("Mean Cl and Cd computed over the most recent 2 complete "
                   "flow-throughs (2 x domain length / freestream speed). "
                   "Less noisy than the live EMA for comparing runs.");
        ImGui::SetWindowFontScale(1.3f);
        ImGui::Text("Cl(a) %+.4f", cla);
        ImGui::Text("Cd(a) %.5f",  cda);
        ImGui::SetWindowFontScale(1.0f);
    }

    // ---- performance + progress block -------------------------------------
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("PERFORMANCE");

    // Processing rate: million lattice-cell updates per second — the honest
    // "how much is the GPU chewing through" number for an LBM solver.
    ImGui::Text("Rate     %.0f MLUPS", r.perf.mlups);
    helpMarker("Million lattice cell updates per second. Every cell of the "
               "grid recomputes every step; this is cells x steps per wall "
               "second.");

    // GPU load (driver query; hidden when the management library is absent)
    // and VRAM pressure on one line.
    if (r.gpuUtilPercent >= 0) {
        ImGui::Text("GPU      %d%%    VRAM  %.0f%%", r.gpuUtilPercent,
                    r.vramUsedFraction * 100.0f);
    } else {
        ImGui::Text("VRAM     %.0f%%", r.vramUsedFraction * 100.0f);
    }

    // Simulated physical time (NOT wall time): steps x dt, shown in ms — a
    // full converged run is typically only a few hundred milliseconds of
    // real airflow.
    ImGui::Text("Sim time %.2f ms", r.simElapsedMs);
    helpMarker("Physical time simulated since the last cold start "
               "(steps x dt). Airflow develops in milliseconds of real time "
               "even when the computation takes minutes.");

    // ETA until the Cl/Cd gate opens. Three states: counting down, done, or
    // no estimate (paused / no timing sample yet).
    if (r.etaSeconds > 0.0) {
        const int total = static_cast<int>(r.etaSeconds + 0.5);
        if (total >= 60) {
            ImGui::Text("ETA      %dm %02ds", total / 60, total % 60);
        } else {
            ImGui::Text("ETA      %ds", total);
        }
        helpMarker("Estimated wall-clock time until the force readouts "
                   "converge (the gate above opens). Based on the measured "
                   "step rate, so it tightens as the run settles.");
    } else if (r.etaSeconds == 0.0) {
        ImGui::TextDisabled("ETA      converged");
    } else {
        ImGui::TextDisabled("ETA      --");
    }

    // Frame rate: ImGui tracks a smoothed FPS from its own delta-time EMA.
    ImGui::Text("FPS      %.0f", ImGui::GetIO().Framerate);

    ImGui::End();
}

// ===========================================================================
// Panel: View (plan 9.2 fifth bullet). Mode toggles (1/2/3), particle pool,
// colormaps, slice positions, screenshot.
// ===========================================================================

void drawViewPanel(UIContext& ctx) {
    UIParams& p = *ctx.params;
    UIReadouts& r = *ctx.readouts;
    UIEvents& ev = *ctx.events;
    if (!ImGui::Begin("View")) { ImGui::End(); return; }

    static const char* kMaps4[] = {"Viridis", "Coolwarm", "Inferno", "Rainbow"};

    ImGui::TextDisabled("MODES");

    // ---- velocity volume: the default hero "wind-tunnel smoke" look ----
    ImGui::Checkbox("Velocity volume  [5]", &p.viz.showVelocityVolume);
    helpMarker("The hero view: a volume raymarch of the air speed. Fast and "
               "wake air glows hot; the quiet freestream fades to a faint haze "
               "you can still see through. Color = speed, no streamlines.");
    if (p.viz.showVelocityVolume) {
        ImGui::Indent();
        int vm = static_cast<int>(p.viz.volumeColormap);
        if (ImGui::Combo("Palette", &vm, kMaps4, 4))
            p.viz.volumeColormap = static_cast<Colormap>(vm);
        // Slow-air haze: floor opacity for undisturbed air. NOT a hard cull —
        // low but non-zero by default so the body stays faintly visible.
        ImGui::SliderFloat("Slow-air opacity", &p.viz.slowAirOpacity, 0.0f, 0.5f,
                           "%.2f");
        helpMarker("How visible the calm, undisturbed air is. 0 = invisible "
                   "(only disturbed air shows), higher = more of a translucent "
                   "fog so you can see structure underneath.");
        ImGui::SliderFloat("Disturbed density", &p.viz.volumeDensity, 0.1f, 1.0f,
                           "%.2f");
        // Which speed maps to the top of the palette (contrast knob).
        ImGui::SliderFloat("Speed scale", &p.viz.velocitySpeedScale, 0.02f, 0.4f,
                           "%.3f");
        // Depth-test escape hatch: ON culls every fog fragment the foil
        // covers on screen (the body pops in front of the smoke); OFF lets
        // the raymarch composite near-side fog over the body like the Q pass.
        ImGui::Checkbox("Foil draws over fog", &p.viz.foilOverVolume);
        helpMarker("When checked, the foil silhouette punches through the fog "
                   "and always reads on top. Unchecked (default), fog between "
                   "the camera and the foil drifts over the body the way the "
                   "vortex skins do.");
        ImGui::Unindent();
    }

    ImGui::Checkbox("Particles  [1]", &p.viz.showParticles);
    ImGui::Checkbox("Slices  [2]", &p.viz.showSlices);
    ImGui::Checkbox("Q-criterion isosurface  [3]", &p.viz.showQRaycast);
    helpMarker("Screen-space volume raycast of the Q-criterion (vortex-core "
               "regions where rotation beats strain). The 3D texture updates "
               "every few frames — this is the screenshot mode.");
    if (p.viz.showQRaycast) {
        ImGui::Indent();
        // Threshold on the NORMALIZED Q in (0,1): the volume texture stores
        // Q/qScale clamped to [0,1], so 1.0 would select nothing.
        ImGui::SliderFloat("Threshold", &p.viz.qThreshold, 0.005f, 0.99f,
                           "%.3f", ImGuiSliderFlags_Logarithmic);
        // Contrast knob: which raw Q maps to 1.0 in the volume (re-uploaded
        // on the renderer's own cadence — no event needed, viz reads
        // VizSettings each frame).
        ImGui::SliderFloat("Q scale", &p.viz.qScale, 1e-6f, 1e-3f, "%.1e",
                           ImGuiSliderFlags_Logarithmic);
        // Color the cores by local speed (shares the velocity volume) instead
        // of the flat cyan tint.
        ImGui::Checkbox("Color by velocity", &p.viz.qColorByVelocity);
        if (p.viz.qColorByVelocity) {
            int qm = static_cast<int>(p.viz.qColormap);
            if (ImGui::Combo("Palette##q", &qm, kMaps4, 4))
                p.viz.qColormap = static_cast<Colormap>(qm);
        }
        ImGui::Unindent();
    }

    ImGui::Spacing();
    ImGui::Checkbox("Foil mesh  [4]", &p.viz.showFoilMesh);
    if (p.viz.showFoilMesh) {
        ImGui::Indent();
        if (ImGui::Checkbox("Voxel view (solver cells)", &p.voxelView))
            ev.voxelViewToggled = true;
        helpMarker("Show the geometry the SOLVER actually sees: the stair-step "
                   "voxelization of the flag field instead of the smooth "
                   "outline. Inside the refinement patch the cubes are half "
                   "size (the 2x fine grid). The definitive way to judge "
                   "whether a VG vane is adequately resolved.");
        ImGui::Checkbox("Wireframe", &p.viz.foilWireframe);
        helpMarker("Draw the airfoil/VG geometry as an edge cage instead of a "
                   "shaded solid — handy for seeing the flow field through the "
                   "body.");
        // Fade applies to both the shaded solid and the wireframe cage. The
        // mesh keeps writing depth at any opacity, so the fog/Q occlusion
        // against the body never changes — only how bright the body reads.
        ImGui::SliderFloat("Opacity", &p.viz.foilOpacity, 0.05f, 1.0f, "%.2f");
        helpMarker("Dims the foil/VG mesh (solid or wireframe) so the flow "
                   "around it stays the star. 1 = fully opaque.");
        ImGui::Unindent();
    }
    if (ImGui::Button("Focus foil  [F]")) ev.frameFoilView = true;

    // ---- particles ----
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("PARTICLES");
    {
        int thousands = p.viz.particleCount / 1000;
        ImGui::SliderInt("Count", &thousands, 100, 2000, "%d k");
        p.viz.particleCount = thousands * 1000;
        // Pool resize re-registers the VBO — release-commit like the geometry
        // sliders so dragging doesn't thrash GL allocations.
        if (ImGui::IsItemDeactivatedAfterEdit()) ev.particleCountChanged = true;
    }
    {
        static const char* kColorBy[] = {"Speed", "Vorticity magnitude"};
        int cb = static_cast<int>(p.viz.particleColorBy);
        if (ImGui::Combo("Color by", &cb, kColorBy, 2))
            p.viz.particleColorBy = static_cast<ParticleColorBy>(cb);
        int cm = static_cast<int>(p.viz.particleColormap);
        if (ImGui::Combo("Colormap", &cm, kMaps4, 4))
            p.viz.particleColormap = static_cast<Colormap>(cm);
        ImGui::SliderFloat("Point size", &p.viz.particlePointSize, 0.5f, 4.0f,
                           "%.1f px");
    }

    // ---- slices: one per axis (plan 9.1 mode 2) ----
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("SLICES");
    static const char* kAxisNames[3] = {"X slice", "Y slice", "Z slice"};
    const int axisMax[3] = {r.dims.nx - 1, r.dims.ny - 1, r.dims.nz - 1};
    for (int i = 0; i < 3; ++i) {
        SliceConfig& sc = p.viz.slices[i];
        ImGui::PushID(i);
        ImGui::Checkbox(kAxisNames[i], &sc.enabled);
        if (sc.enabled) {
            ImGui::Indent();
            ImGui::SliderInt("Position", &sc.cell, 0, std::max(axisMax[i], 1));
            static const char* kFields[] = {"|u| speed", "Vorticity z", "Pressure"};
            int f = static_cast<int>(sc.field);
            if (ImGui::Combo("Field", &f, kFields, 3)) {
                sc.field = static_cast<SliceField>(f);
                // Field choice implies the honest colormap pairing (plan 9.1):
                // sequential for magnitudes, diverging for signed fields.
                sc.colormap = (sc.field == SliceField::SpeedMag)
                                  ? Colormap::Viridis : Colormap::Coolwarm;
            }
            ImGui::Unindent();
        }
        ImGui::PopID();
    }

    ImGui::Spacing();
    ImGui::Separator();
    if (ImGui::Button("Screenshot (PNG)", ImVec2(-1, 0))) ev.screenshot = true;
    ImGui::End();
}

// ===========================================================================
// Panel: Mesh — the two-level refinement patch (plan M-refine) + the mesh-
// sequencing startup toggle. The patch is a 2x-finer nested lattice over the
// foil/VG/near-wake region, coupled to the base grid every step; the panel
// configures its margins, shows the live patch box in a mini domain diagram,
// and surfaces the VRAM bill BEFORE the user commits.
// ===========================================================================

/// @brief Mini top-view of the domain with the refinement patch box drawn
/// over a foil silhouette — the at-a-glance "what am I refining" picture.
static void drawPatchDiagram(const UIReadouts& r, bool enabled,
                             const ImVec2& canvasPos, const ImVec2& canvasSize) {
    ImDrawList* dl = ImGui::GetWindowDrawList();

    const float W  = canvasSize.x;
    const float H  = canvasSize.y;
    const float px = canvasPos.x;
    const float py = canvasPos.y;

    // Background frame (domain extents).
    dl->AddRectFilled(ImVec2(px, py), ImVec2(px + W, py + H),
                      IM_COL32(18, 20, 25, 255), 4.0f);
    dl->AddRect(ImVec2(px, py), ImVec2(px + W, py + H),
                IM_COL32(50, 55, 65, 200), 4.0f);

    // Foil silhouette: anchored like the real layout (quarter-chord at 30%
    // of nx, mid-height, chord = nx/3).
    const float chordPx = W / 3.0f;
    const float foilX0  = px + 0.30f * W - 0.25f * chordPx; // LE
    const float foilX1  = foilX0 + chordPx;                 // TE
    const float foilMY  = py + H * 0.5f;
    {
        const int segs = 32;
        std::vector<ImVec2> upper, lower;
        upper.reserve(segs + 1);
        lower.reserve(segs + 1);
        for (int s = 0; s <= segs; ++s) {
            const float t   = static_cast<float>(s) / static_cast<float>(segs);
            const float xpt = foilX0 + t * (foilX1 - foilX0);
            const float tck = H * 0.12f * std::sqrt(t) * (1.0f - t) * 2.5f;
            const float cam = H * 0.06f * (4.0f * t * (1.0f - t));
            upper.push_back(ImVec2(xpt, foilMY - cam - tck));
            lower.push_back(ImVec2(xpt, foilMY - cam + tck));
        }
        dl->AddPolyline(upper.data(), segs + 1, IM_COL32(220, 90, 90, 220), 0, 1.5f);
        dl->AddPolyline(lower.data(), segs + 1, IM_COL32(220, 90, 90, 220), 0, 1.5f);
    }

    if (!enabled || r.dims.nx <= 0) {
        const char* lbl = enabled ? "patch pending re-init" : "uniform grid";
        const ImVec2 ts = ImGui::CalcTextSize(lbl);
        dl->AddText(ImVec2(px + (W - ts.x) * 0.5f, py + H - ts.y - 4.0f),
                    IM_COL32(110, 115, 125, 200), lbl);
        return;
    }

    // Patch box mapped from coarse cells into diagram space. When the sim has
    // a live patch, draw the REAL box; otherwise nothing (pending re-init).
    if (r.refine.active) {
        const float sx = W / static_cast<float>(r.dims.nx);
        const float sy = H / static_cast<float>(r.dims.ny);
        // y is flipped: lattice y=0 is the bottom slip wall.
        const float bx0 = px + r.refine.patchX0 * sx;
        const float bx1 = px + r.refine.patchX1 * sx;
        const float by0 = py + H - r.refine.patchY1 * sy;
        const float by1 = py + H - r.refine.patchY0 * sy;
        dl->AddRectFilled(ImVec2(bx0, by0), ImVec2(bx1, by1),
                          IM_COL32(66, 140, 245, 28), 2.0f);
        dl->AddRect(ImVec2(bx0, by0), ImVec2(bx1, by1),
                    IM_COL32(66, 140, 245, 220), 2.0f, 0, 1.5f);
        const char* lbl = "2x patch";
        dl->AddText(ImVec2(bx0 + 4.0f, by0 + 2.0f),
                    IM_COL32(140, 185, 255, 230), lbl);
    }
}

/// @brief Draw the Mesh panel: refinement patch config + startup options.
void drawMeshPanel(UIContext& ctx) {
    UIParams&   p  = *ctx.params;
    UIReadouts& r  = *ctx.readouts;
    UIEvents&   ev = *ctx.events;

    if (!ImGui::Begin("Mesh")) { ImGui::End(); return; }

    // ---- refinement patch ----
    ImGui::TextDisabled("REFINEMENT PATCH");
    helpMarker(
        "Two-level grid refinement: a 2x-finer nested lattice covering the "
        "foil, VGs, and near wake, two-way coupled to the base grid every "
        "step. Doubles the wall/VG/boundary-layer resolution exactly where "
        "the guidance reads, at a fraction of the cost of doubling the whole "
        "domain. Effective Re is shared across levels — the patch buys "
        "RESOLUTION at the same Re, not a higher Re.");

    bool enabled = p.refine.enabled;
    if (ImGui::Checkbox("Enable 2x refinement patch", &enabled)) {
        p.refine.enabled = enabled;
        ev.meshRefinementChanged = true;
    }

    // ---- live domain diagram ----
    ImGui::Spacing();
    {
        const ImVec2 cpos = ImGui::GetCursorScreenPos();
        const float W = ImGui::GetContentRegionAvail().x;
        constexpr float H = 64.0f;
        drawPatchDiagram(r, p.refine.enabled, cpos, ImVec2(W, H));
        ImGui::Dummy(ImVec2(W, H));
    }

    if (p.refine.enabled) {
        // Margins around the solid bbox, in chord fractions. Release-commit:
        // every change re-derives the patch and re-allocates the fine level.
        ImGui::Spacing();
        ImGui::TextDisabled("MARGINS (chords around the foil+VG bbox)");
        auto marginSlider = [&ev](const char* label, float* v, float lo,
                                  float hi, const char* tip) {
            ImGui::SliderFloat(label, v, lo, hi, "%.2f c");
            if (ImGui::IsItemDeactivatedAfterEdit())
                ev.meshRefinementChanged = true;
            helpMarker(tip);
        };
        marginSlider("Upstream", &p.refine.upstreamC, 0.05f, 0.50f,
                     "Patch extent ahead of the leading edge — covers the "
                     "stagnation/suction-peak region.");
        marginSlider("Wake", &p.refine.wakeC, 0.10f, 1.00f,
                     "Patch extent behind the trailing edge — covers the "
                     "near-wake rollup of the VG vortices.");
        marginSlider("Above", &p.refine.aboveC, 0.05f, 0.50f,
                     "Patch extent above the foil — must cover the suction-"
                     "surface boundary layer plus the VG vanes and their "
                     "shed vortex path.");
        marginSlider("Below", &p.refine.belowC, 0.05f, 0.50f,
                     "Patch extent below the foil (pressure side needs less).");

        // ---- live status from the solver ----
        ImGui::Spacing();
        ImGui::Separator();
        if (r.refine.active) {
            ImGui::TextDisabled(
                "fine grid %d x %d x %d  (~%.1f GB VRAM)",
                r.refine.fineDims.nx, r.refine.fineDims.ny, r.refine.fineDims.nz,
                r.refine.fineVramGB);
            if (r.refine.forcesFromFine) {
                ImGui::TextColored(kColGood, "forces measured on the 2x grid");
                helpMarker("The patch covers every solid cell, so the "
                           "momentum-exchange force reduction runs at 2x wall "
                           "resolution — the patch's whole purpose.");
            } else {
                ImGui::TextColored(kColWarn,
                                   "solids extend past the patch — forces on "
                                   "base grid");
                helpMarker("Increase the margins until the foil and every VG "
                           "sit inside the patch to get 2x-resolution "
                           "forces.");
            }
        } else {
            ImGui::TextColored(kColWarn, "patch inactive (init failed or "
                                         "pending re-init)");
        }

        // VRAM guard readout: warn when the whole allocation crowds the card.
        if (r.vramUsedFraction > 0.8f) {
            ImGui::TextColored(kColBad, "VRAM %.0f%% — consider disabling the "
                                        "patch or a smaller preset",
                               r.vramUsedFraction * 100.0f);
        }
    }

    // ---- startup: mesh-sequencing pre-convergence ----
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("STARTUP");
    ImGui::Checkbox("Pre-converge at 4x coarse", &p.preconvergeCoarse);
    helpMarker(
        "Mesh sequencing: every cold start first converges a 4x-coarser "
        "companion sim (seconds — its grid is 1/64 the cells), then upsamples "
        "that developed flow onto the full grid and continues. The wake and "
        "circulation arrive almost immediately instead of convecting in from "
        "scratch, so the Cl/Cd gate opens much sooner. The settle window "
        "still applies — readouts stay gated until the fine-grid flow has "
        "re-adjusted.");
    if (r.preconvergeProgress >= 0.0f) {
        ImGui::ProgressBar(r.preconvergeProgress, ImVec2(-1, 0));
        ImGui::TextDisabled("pre-converging coarse companion...");
    }

    ImGui::End();
}

// ===========================================================================
// STL import modal (plan 7.2): bounding box, axis remap presets, chord-cell
// scale, and the plan 7.4 z-boundary mode toggle. main.cpp opens it after a
// successful loadStl(); Import/Cancel raise events.
// ===========================================================================

void drawStlImportModal(UIContext& ctx) {
    UIParams& p = *ctx.params;
    UIEvents& ev = *ctx.events;
    StlImportUI& s = p.stlImport;
    if (!s.open) return;

    ImGui::OpenPopup("Import STL");
    ImGui::SetNextWindowSize(ImVec2(460, 0), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal("Import STL", nullptr,
                                ImGuiWindowFlags_AlwaysAutoResize)) {
        return;
    }

    ImGui::Text("%s", s.fileName.c_str());
    ImGui::TextDisabled("%s  -  %u triangles  -  %s", s.solidName.c_str(),
                        s.triangleCount, s.wasBinary ? "binary" : "ASCII");
    const Vec3f ext = s.bounds.size();
    ImGui::Text("Bounds: %.3g x %.3g x %.3g (file units)", ext.x, ext.y, ext.z);
    ImGui::Spacing();
    ImGui::Separator();

    // ---- axis remap presets (plan 7.2: STLs arrive in random orientation) --
    static const char* kAxisPresets[] = {
        "XYZ as-is (x = chord, y = up, z = span)",
        "Z-up CAD  (file Z becomes up)",
        "Y-up span (file Y becomes span)"};
    int axisIdx = static_cast<int>(s.axisPreset);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##axis", &axisIdx, kAxisPresets, 3))
        s.axisPreset = static_cast<StlAxisPreset>(axisIdx);

    // ---- uniform scale: longest x-extent -> chosen chord in cells ----
    ImGui::SliderInt("Chord (cells)", &s.chordCells, 32, 512);
    helpMarker("The mesh is uniformly scaled so its streamwise extent spans "
               "this many lattice cells, then centered at the standard foil "
               "position. Match the grid's chord resolution unless the object "
               "is intentionally smaller than a full chord.");

    // ---- z-boundary mode (plan 7.4) ----
    ImGui::Spacing();
    ImGui::TextDisabled("SPANWISE (z) BOUNDARY");
    int zMode = s.zFreeSlipWalls ? 1 : 0;
    ImGui::RadioButton("Periodic — 2.5D section (airfoil-like geometry)",
                       &zMode, 0);
    ImGui::RadioButton("Free-slip walls — full 3D object", &zMode, 1);
    s.zFreeSlipWalls = (zMode == 1);
    helpMarker("Spanwise-periodic boundaries pretend the geometry repeats "
               "forever along the span — right for wing sections, wrong for "
               "a finite 3D body. Free-slip walls box the object in without "
               "adding wall boundary layers.");

    ImGui::Spacing();
    ImGui::TextColored(kColWarn, "Non-watertight meshes voxelize badly —\n"
                                 "you will be warned if parity errors appear.");
    ImGui::Separator();

    if (ImGui::Button("Import", ImVec2(140, 0))) {
        ev.stlImportConfirmed = true;
        s.open = false;
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(140, 0))) {
        ev.stlImportCancelled = true;
        s.open = false;
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

/// @brief Bottom-left status overlay: transient messages (load errors, cache
/// notes) without dedicating a docked panel to one line of text.
// ===========================================================================
// Speed-scale legend: a vertical colorbar at the right edge of the render
// area mapping the active speed palette to physical airspeed (in the user's
// chosen unit). On by default; collapses to a small re-show chip when hidden.
// The palette evaluators below mirror the GLSL/CUDA versions exactly so the
// legend colors are the ones actually on screen.
// ===========================================================================

/// One palette sample as an ImGui color. Coefficients/structure match the
/// shader-side viridis/inferno polynomial fits, the two-segment coolwarm,
/// and the HSV-sweep rainbow (selector order 0/1/2/3 — Colormap enum).
ImU32 legendPaletteColor(Colormap map, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    float r, g, b;
    auto poly6 = [t](const float c[7][3], float* r, float* g, float* b) {
        float acc[3];
        for (int ch = 0; ch < 3; ++ch) {
            float v = c[6][ch];
            for (int k = 5; k >= 0; --k) v = v * t + c[k][ch];
            acc[ch] = std::clamp(v, 0.0f, 1.0f);
        }
        *r = acc[0]; *g = acc[1]; *b = acc[2];
    };
    switch (map) {
        case Colormap::Rainbow: {
            // HSV hue sweep blue -> green -> red, S = V = 1.
            const float h = (1.0f - t) * 4.0f;
            auto chan = [h](float off) {
                const float k = std::fmod(off + h, 6.0f);
                return 1.0f - std::max(0.0f, std::min({k, 4.0f - k, 1.0f}));
            };
            r = chan(5.0f); g = chan(3.0f); b = chan(1.0f);
            break;
        }
        case Colormap::Inferno: {
            static const float c[7][3] = {
                {-0.00021f,  0.00016f, -0.01976f}, { 0.10677f,  0.56329f,  3.93245f},
                {11.60249f, -3.97282f, -15.94254f}, {-41.70399f, 17.43639f, 44.35414f},
                {77.16296f, -33.40235f, -81.80730f}, {-71.31899f, 32.62606f, 73.20951f},
                {25.13112f, -12.24266f, -23.07032f}};
            poly6(c, &r, &g, &b);
            break;
        }
        case Colormap::Coolwarm: {
            const float blue[3]  = {0.230f, 0.299f, 0.754f};
            const float white[3] = {0.865f, 0.865f, 0.865f};
            const float red[3]   = {0.706f, 0.016f, 0.150f};
            const float s = (t < 0.5f) ? t * 2.0f : (t - 0.5f) * 2.0f;
            const float e = s * s * (3.0f - 2.0f * s); // smoothstep
            const float* a = (t < 0.5f) ? blue : white;
            const float* c = (t < 0.5f) ? white : red;
            r = a[0] + (c[0] - a[0]) * e;
            g = a[1] + (c[1] - a[1]) * e;
            b = a[2] + (c[2] - a[2]) * e;
            break;
        }
        case Colormap::Viridis:
        default: {
            static const float c[7][3] = {
                { 0.2777273f,  0.0054073f,  0.3340998f}, { 0.1050930f, 1.4046135f, 1.3845902f},
                {-0.3308618f,  0.2148476f,  0.0950952f}, {-4.6342305f, -5.7991010f, -19.3324410f},
                { 6.2282699f, 14.1799334f, 56.6905526f}, { 4.7763850f, -13.7451454f, -65.3530326f},
                {-5.4354559f,  4.6458526f, 26.3124352f}};
            poly6(c, &r, &g, &b);
            break;
        }
    }
    return IM_COL32(static_cast<int>(r * 255.0f + 0.5f),
                    static_cast<int>(g * 255.0f + 0.5f),
                    static_cast<int>(b * 255.0f + 0.5f), 255);
}

void drawVelocityLegend(UIContext& ctx) {
    UIParams& p = *ctx.params;

    // Only meaningful while something on screen is speed-colored. The Q
    // isosurface takes palette precedence over the fog volume (it draws on
    // top and is the default hero mode).
    const bool qActive   = p.viz.showQRaycast && p.viz.qColorByVelocity;
    const bool volActive = p.viz.showVelocityVolume;
    if (!qActive && !volActive) return;

    // Top-of-scale speed: the lattice normalization converted to physical
    // units, then to the user's display unit. Guard the unscaled-startup
    // frame (dt = 0 would NaN the labels).
    const LatticeScaling& sc = ctx.readouts->scaling;
    if (!(sc.dt > 0.0f) || !(sc.dx > 0.0f)) return;
    const float unitPerMs = speedUnitPerMs(p.speedUnit);
    const float topSpeed =
        sc.velocityToPhysical(p.viz.velocitySpeedScale) * unitPerMs;

    const ImGuiWindowFlags overlayFlags =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings;

    // Anchor: vertically centered on the render area's right edge.
    const ImVec2 anchor(gRenderAreaPos.x + gRenderAreaSize.x - 12.0f,
                        gRenderAreaPos.y + gRenderAreaSize.y * 0.5f);
    ImGui::SetNextWindowPos(anchor, ImGuiCond_Always, ImVec2(1.0f, 0.5f));
    ImGui::SetNextWindowBgAlpha(p.showVelocityLegend ? 0.55f : 0.35f);

    // ---- collapsed: a small chip that brings the legend back ----
    if (!p.showVelocityLegend) {
        ImGui::Begin("##speedLegendChip", nullptr, overlayFlags);
        if (ImGui::SmallButton("<##showLegend")) p.showVelocityLegend = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show speed scale");
        ImGui::End();
        return;
    }

    ImGui::Begin("##speedLegend", nullptr, overlayFlags);

    // Header: caption + the hide chip on the same row.
    ImGui::TextDisabled("SPEED");
    ImGui::SameLine();
    if (ImGui::SmallButton(">##hideLegend")) p.showVelocityLegend = false;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Hide speed scale");

    const Colormap map = qActive ? p.viz.qColormap : p.viz.volumeColormap;

    // ---- gradient bar: stacked vertically-interpolated strips. Fast (the
    // palette top) is at the TOP of the bar, zero at the bottom. ----
    const float barW = 18.0f, barH = 200.0f;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const int strips = 48;
    for (int i = 0; i < strips; ++i) {
        const float tTop = 1.0f - static_cast<float>(i) / strips;
        const float tBot = 1.0f - static_cast<float>(i + 1) / strips;
        const float y0 = p0.y + barH * static_cast<float>(i) / strips;
        const float y1 = p0.y + barH * static_cast<float>(i + 1) / strips;
        const ImU32 cTop = legendPaletteColor(map, tTop);
        const ImU32 cBot = legendPaletteColor(map, tBot);
        dl->AddRectFilledMultiColor(ImVec2(p0.x, y0), ImVec2(p0.x + barW, y1),
                                    cTop, cTop, cBot, cBot);
    }
    dl->AddRect(p0, ImVec2(p0.x + barW, p0.y + barH), IM_COL32(255, 255, 255, 60));

    // ---- tick labels: five even stops from top-of-scale down to zero ----
    float labelMaxW = 0.0f;
    for (int i = 0; i <= 4; ++i) {
        const float frac = 1.0f - static_cast<float>(i) / 4.0f;
        const float y = p0.y + barH * static_cast<float>(i) / 4.0f;
        char label[32];
        std::snprintf(label, sizeof label, "%.0f %s", topSpeed * frac,
                      speedUnitLabel(p.speedUnit));
        dl->AddText(ImVec2(p0.x + barW + 6.0f, y - 7.0f),
                    IM_COL32(220, 225, 235, 255), label);
        labelMaxW = std::max(labelMaxW, ImGui::CalcTextSize(label).x);
    }

    // Reserve the drawn region in the layout so auto-resize fits the bar.
    ImGui::Dummy(ImVec2(barW + 6.0f + labelMaxW, barH));
    ImGui::End();
}

// ===========================================================================
// Divergence modal: a centered, unmissable popup when the NaN watchdog trips.
// The in-panel red box still exists, but it lives in the Sim panel which can
// be hidden behind another tab — a silently frozen sim looked like a bug.
// ===========================================================================

void drawDivergenceModal(UIContext& ctx) {
    UIReadouts& r = *ctx.readouts;
    UIEvents& ev = *ctx.events;

    // Rising edge -> open exactly once per latch (a cold reset clears the
    // watchdog latch, re-arming this for any future trip).
    static bool wasTripped = false;
    if (r.nanTripped && !wasTripped) ImGui::OpenPopup("Simulation diverged");
    wasTripped = r.nanTripped;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.5f,
                                   vp->WorkPos.y + vp->WorkSize.y * 0.5f),
                            ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(480.0f, 0.0f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal("Simulation diverged", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextColored(kColBad, "The flow field hit a numeric blow-up (NaN).");
        ImGui::TextWrapped("%s", r.nanDiagnosis.c_str());
        ImGui::Spacing();
        if (ImGui::Button("Restart simulation", ImVec2(160, 0))) {
            ev.resetCold = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Keep paused", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void drawStatusOverlay(const UIContext& ctx) {
    if (ctx.statusMessage.empty()) return;
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + 12.0f,
                                   vp->WorkPos.y + vp->WorkSize.y - 12.0f),
                            ImGuiCond_Always, ImVec2(0.0f, 1.0f));
    ImGui::SetNextWindowBgAlpha(0.65f);
    ImGui::Begin("##status", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_AlwaysAutoResize |
                 ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
                 ImGuiWindowFlags_NoDocking);
    ImGui::TextUnformatted(ctx.statusMessage.c_str());
    ImGui::End();
}

} // namespace

// ===========================================================================
// Public entry points (ui.h contract).
// ===========================================================================

bool uiInit(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    // Docking branch feature: panels dock against the window edges.
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::GetIO().ConfigWindowsMoveFromTitleBarOnly = true;
    applyFoilStyle();
    if (!ImGui_ImplGlfw_InitForOpenGL(window, /*install_callbacks=*/true)) return false;
    if (!ImGui_ImplOpenGL3_Init("#version 460")) return false;
    return true;
}

void uiShutdown() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void uiBeginFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void drawUI(UIContext& ctx) {
    // Defensive: the frame loop always supplies all three, but a null deref
    // inside ImGui would be miserable to trace back here.
    if (!ctx.params || !ctx.readouts || !ctx.events) return;

    beginDockspace();
    handleHotkeys(ctx);

    drawAirfoilPanel(ctx);
    drawVGEditorPanel(ctx);
    drawVGGuidancePanel(ctx);
    drawSimPanel(ctx);
    drawReadoutsPanel(ctx);
    drawViewPanel(ctx);
    drawMeshPanel(ctx);
    drawStlImportModal(ctx);
    drawDivergenceModal(ctx);
    drawVelocityLegend(ctx);
    drawStatusOverlay(ctx);
}

void uiEndFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool uiWantsInput() {
    const ImGuiIO& io = ImGui::GetIO();
    return io.WantCaptureMouse || io.WantCaptureKeyboard;
}

} // namespace foilcfd
