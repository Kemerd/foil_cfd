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
    ImGui::DockBuilderDockWindow("Readouts", rightBottom);
    ImGui::DockBuilderFinish(dockspaceId);
}

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
    helpMarker("Changing airspeed rescales the units and restarts the startup "
               "ramp, but the cached flow stays valid (plan: airspeed is not "
               "part of the snapshot key). The kn/mph/m-s buttons only change "
               "the display unit — the simulation always runs in SI.");

    // Chord is a pure UNITS rescale, exactly like airspeed: grid dims, u_lat,
    // and the flag field are untouched (only dx/dt/Re-target change), so it
    // rides the cheap scaling-changed path — a full re-init would needlessly
    // cold-start the flow and destroy the warm cache on every Re-sweep tick.
    ImGui::SliderFloat("Chord", &p.chordM, 0.2f, 4.0f, "%.2f m");
    if (ImGui::IsItemDeactivatedAfterEdit()) ev.airspeedChanged = true;
    helpMarker("Physical chord length. Like airspeed, this only rescales the "
               "unit conversion (target Reynolds number) — the cached flow "
               "stays valid.");

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
bool drawVGEntry(VGParams& vg, int chordCells) {
    bool edited = false;
    // Type combo — discrete, commits immediately.
    static const char* kTypeNames[] = {"Single vane", "Counter-rotating pair",
                                       "Co-rotating array", "Ramp"};
    int typeIdx = static_cast<int>(vg.type);
    if (ImGui::Combo("Type", &typeIdx, kTypeNames, 4)) {
        vg.type = static_cast<VGType>(typeIdx);
        edited = true;
    }

    // Continuous params: commit on release only (plan 13 slider rule).
    auto releaseSlider = [&edited](const char* label, float* v, float lo,
                                   float hi, const char* fmt) {
        ImGui::SliderFloat(label, v, lo, hi, fmt);
        if (ImGui::IsItemDeactivatedAfterEdit()) edited = true;
    };
    releaseSlider("Station x/c", &vg.x_c, 0.01f, 0.40f, "%.3f");
    releaseSlider("Height h/c", &vg.height_c, 0.002f, 0.030f, "%.4f");
    releaseSlider("Length (h)", &vg.length_h, 1.0f, 6.0f, "%.1f");
    releaseSlider("Incidence", &vg.beta_deg, -30.0f, 30.0f, "%.1f deg");
    const bool multiUnit = (vg.type == VGType::CounterRotatingPair
                            || vg.type == VGType::CoRotatingArray);
    if (multiUnit) {
        releaseSlider("Pitch (c)", &vg.pitch_c, 0.01f, 0.20f, "%.3f");
    }
    if (vg.type == VGType::CounterRotatingPair) {
        releaseSlider("Gap (h)", &vg.gap_h, 1.0f, 6.0f, "%.1f");
        bool cfd = vg.commonFlowDown;
        if (ImGui::Checkbox("Common-flow-down", &cfd)) {
            vg.commonFlowDown = cfd;
            edited = true;
        }
    }
    if (multiUnit) {
        int count = vg.count;
        if (ImGui::SliderInt("Units", &count, 1, 16)) { /* drag preview */ }
        if (ImGui::IsItemDeactivatedAfterEdit()) edited = true;
        vg.count = count;
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
    const int chordCells = ctx.readouts->scaling.chordCells;
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

    ImGui::TextDisabled("Edits restart WARM from the cached clean-foil flow.");
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
            if (drawVGEntry(p.vgs[static_cast<size_t>(i)], chordCells)) {
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

    if (ImGui::Button("Save clean state")) ev.saveCleanState = true;
    ImGui::SameLine();
    ImGui::Checkbox("Auto-cache", &p.autoCacheClean);
    helpMarker("The converged clean-foil flow is the expensive part. Caching "
               "it (VRAM + disk) lets VG edits warm-restart in seconds "
               "instead of recomputing the wing. Auto-cache captures it as "
               "soon as the clean flow converges.");

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
        const float cl = smoothValue(ImGui::GetID("cl"), r.forces.cl);
        const float cd = smoothValue(ImGui::GetID("cd"), r.forces.cd);
        const float ld = smoothValue(ImGui::GetID("ld"), r.forces.liftToDrag);
        // Large-type readout: 2x font scale on the three hero numbers.
        ImGui::SetWindowFontScale(1.6f);
        ImGui::Text("Cl  %+.4f", cl);
        ImGui::Text("Cd  %.5f", cd);
        ImGui::Text("L/D %.2f", ld);
        ImGui::SetWindowFontScale(1.0f);
    }
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

    static const char* kMaps3[] = {"Viridis", "Coolwarm", "Inferno"};

    ImGui::TextDisabled("MODES");

    // ---- velocity volume: the default hero "wind-tunnel smoke" look ----
    ImGui::Checkbox("Velocity volume  [5]", &p.viz.showVelocityVolume);
    helpMarker("The hero view: a volume raymarch of the air speed. Fast and "
               "wake air glows hot; the quiet freestream fades to a faint haze "
               "you can still see through. Color = speed, no streamlines.");
    if (p.viz.showVelocityVolume) {
        ImGui::Indent();
        int vm = static_cast<int>(p.viz.volumeColormap);
        if (ImGui::Combo("Palette", &vm, kMaps3, 3))
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
        ImGui::Unindent();
    }

    ImGui::Spacing();
    ImGui::Checkbox("Foil mesh  [4]", &p.viz.showFoilMesh);
    if (p.viz.showFoilMesh) {
        ImGui::Indent();
        ImGui::Checkbox("Wireframe", &p.viz.foilWireframe);
        helpMarker("Draw the airfoil/VG geometry as an edge cage instead of a "
                   "shaded solid — handy for seeing the flow field through the "
                   "body.");
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
        if (ImGui::Combo("Colormap", &cm, kMaps3, 3))
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
    drawStlImportModal(ctx);
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
