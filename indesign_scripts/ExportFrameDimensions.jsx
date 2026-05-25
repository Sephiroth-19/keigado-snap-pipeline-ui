/*
 * ExportFrameDimensions.jsx  v3 — Frame measurer + guide reader
 * ==============================================================
 *
 * v3 FIXES:
 *   1. getAllGuideYsOnLayer() now records the CENTRE of each thin rectangle
 *      instead of both top and bottom edges.  This means each physical guide
 *      line produces exactly ONE Y value, not two.
 *   2. getGuideRatiosForFrame() collapses any remaining nearby values
 *      (within 8mm) as a safety net before selecting top/bottom guides.
 *   3. deriveScaleClamp() now uses min=100 always.  The old floor of 125–127
 *      was clamping Python's per-person scale calculations and preventing
 *      correct face placement.
 *
 * RESULT: guide_ratios in frame_config.json are now consistent across all
 * student frames and match the actual visual guide positions in the template.
 * Python can use bottom_ratio directly as target_chin_y with no manual override.
 *   Run ONCE on any InDesign template before face_offset_calculator.py.
 *   Extracts everything Python needs to position faces correctly for THIS
 *   specific template — frame sizes AND the red guide positions that define
 *   where faces should land.
 *
 * WHAT IT MEASURES
 *   1. Student portrait frame  W × H (mm)
 *   2. Teacher portrait frame  W × H (mm)
 *   3. The two horizontal red guides (顔位置ガイド layer) that bracket the
 *      face zone, expressed as ratios of the student frame height:
 *        guide_top_ratio   → becomes target_forehead_y  (top of face zone)
 *        guide_bottom_ratio → becomes target_chin_y     (where chin should land)
 *   4. A scale clamp range derived from the frame aspect ratio, so faces
 *      are never over- or under-zoomed for this layout's photo crop style.
 *
 * HOW THE GUIDE RATIOS WORK
 *   The red guides are horizontal lines on the 顔位置ガイド layer.
 *   For each student frame, we find which guides overlap it and compute:
 *     guide_top_ratio    = (guide_y - frame_top) / frame_height
 *     guide_bottom_ratio = (guide_y - frame_top) / frame_height
 *   Python uses guide_bottom_ratio as target_chin_y directly.
 *   The face height (forehead→chin) as a fraction of frame height becomes
 *   the basis for the scale clamp range.
 *
 * USAGE
 *   1. Open your InDesign template
 *   2. Edit CONFIG.outputFile below
 *   3. Run via Scripts panel
 *   4. Run face_offset_calculator.py — it reads frame_config.json automatically
 *
 * OUTPUT  frame_config.json  (place next to manifest.json)
 */

#target indesign

// ══════════════════════════════════════════════════════════════════
// CONFIGURATION
// ══════════════════════════════════════════════════════════════════

var CONFIG = {
    // ← Edit this to your project folder (same folder as manifest.json)
    outputFile: "D:\\career\\1_LTID\\Photography\\output_千早高_FINAL_v6\\frame_config.json",

    // Layer names — must match AutoPlacePhotos CONFIG.layers
    portraitLayer:    "Default",
    portraitLayerAlt: "本番カット",
    guideLayer:       "顔位置ガイド",   // layer that holds the two red face-zone guides

    // Frames whose InDesign object name STARTS WITH this prefix are treated
    // as teacher slots — regardless of size. This matches the AutoPlace
    // convention ("T_Photo_01", "T_Photo_02", …). Without this we fall back
    // to "teacher = frame whose size differs from students", which fails
    // when teacher and student frames happen to be the same dimensions.
    teacherFrameNamePrefix: "T_Photo",

    // Which spread to measure (−1 = active spread)
    spreadIndex: -1,

    // How much tolerance (mm) when deciding two guide lines are "near" a frame
    guideTolerance: 5
};


// ══════════════════════════════════════════════════════════════════
// UNIT HELPER — switch doc to mm, read, restore
// ══════════════════════════════════════════════════════════════════

function withMM(doc, fn) {
    var origH = null, origV = null;
    try {
        origH = doc.viewPreferences.horizontalMeasurementUnits;
        origV = doc.viewPreferences.verticalMeasurementUnits;
        doc.viewPreferences.horizontalMeasurementUnits = MeasurementUnits.MILLIMETERS;
        doc.viewPreferences.verticalMeasurementUnits   = MeasurementUnits.MILLIMETERS;
    } catch(e) {}
    var result = fn();
    try {
        if (origH !== null) doc.viewPreferences.horizontalMeasurementUnits = origH;
        if (origV !== null) doc.viewPreferences.verticalMeasurementUnits   = origV;
    } catch(e) {}
    return result;
}

function getDocObj(item) {
    var d = item;
    while (d && d.constructor.name !== "Document") {
        try { d = d.parent; } catch(e) { return null; }
    }
    return d;
}


// ══════════════════════════════════════════════════════════════════
// FRAME HELPERS
// ══════════════════════════════════════════════════════════════════

function getAllPageItems(container) {
    var items = [];
    var top = container.allPageItems;
    for (var i = 0; i < top.length; i++) { items.push(top[i]); }
    return items;
}

function getFramesOnLayer(spread, layerName, altLayerName) {
    var frames = [];
    var allItems = getAllPageItems(spread);
    for (var i = 0; i < allItems.length; i++) {
        var item = allItems[i];
        if (item.constructor.name !== "Rectangle") continue;
        try {
            var lName = item.itemLayer.name;
            if (lName === layerName || (altLayerName && lName === altLayerName)) {
                frames.push(item);
            }
        } catch (e) {}
    }
    return frames;
}

function findLargestFrame(frames) {
    var maxArea = 0, idx = -1;
    for (var i = 0; i < frames.length; i++) {
        var gb = frames[i].geometricBounds;
        var area = (gb[3] - gb[1]) * (gb[2] - gb[0]);
        if (area > maxArea) { maxArea = area; idx = i; }
    }
    return idx;
}

function frameBoundsMM(frame, doc) {
    return withMM(doc, function() {
        var gb = frame.geometricBounds; // [top, left, bottom, right]
        return {
            top:    gb[0], left:  gb[1],
            bottom: gb[2], right: gb[3],
            w: Math.abs(gb[3] - gb[1]),
            h: Math.abs(gb[2] - gb[0])
        };
    });
}


// ══════════════════════════════════════════════════════════════════
// GUIDE READER
// Reads horizontal guide lines on a specific layer (as rectangles
// used as visual guides), OR InDesign ruler guides, OR thin
// rectangles — handles all three conventions designers use.
// ══════════════════════════════════════════════════════════════════

function getGuideYPositionsMM(spread, doc, guideLayerName, tolerance, forTeacher) {
    /*
     * Strategy A: Look for very thin rectangles on the guide layer
     *   (common: designers draw a 0.25pt hairline rectangle as a guide)
     * Strategy B: Look for InDesign ruler guides (doc.guides)
     * Returns array of Y positions in mm, sorted top→bottom.
     */
    var yPositions = [];

    // ── Strategy A: thin rectangles on the guide layer ──────────
    var allItems = getAllPageItems(spread);
    for (var i = 0; i < allItems.length; i++) {
        var item = allItems[i];
        if (item.constructor.name !== "Rectangle" &&
            item.constructor.name !== "GraphicLine") continue;
        try {
            if (item.itemLayer.name !== guideLayerName) continue;
        } catch(e) { continue; }

        // Check if item is inside a group
        var isGrouped = (item.parent && 
                         item.parent.constructor.name === "Group");

        // Teacher guides = top-level lines (not in a group)
        // Student guides = lines inside a group
        if (forTeacher && isGrouped) continue;
        if (!forTeacher && !isGrouped) continue;

        var b = withMM(doc, function() { return item.geometricBounds; });
        var itemH = Math.abs(b[2] - b[0]);
        var centerY = (b[0] + b[2]) / 2;

        if (itemH < 3 || item.constructor.name === "GraphicLine") {
            yPositions.push(Math.round(centerY * 100) / 100);
        }
        // Also accept wide, nearly-zero-height rectangles used as rules
        else if (itemH < 1.5 && itemW > 10) {
            yPositions.push(Math.round(centerY * 100) / 100);
        }
    }

    // ── Strategy B: InDesign ruler guides ────────────────────────
    // (Ruler guides aren't on layers — check all guides on the spread)
    try {
        var guides = spread.guides;
        for (var g = 0; g < guides.length; g++) {
            try {
                if (guides[g].orientation === HorizontalOrVertical.HORIZONTAL) {
                    var gy = withMM(doc, function() { return guides[g].location; });
                    yPositions.push(Math.round(gy * 100) / 100);
                }
            } catch(e) {}
        }
    } catch(e) {}

    // ── Deduplicate positions within tolerance ───────────────────
    yPositions.sort(function(a, b) { return a - b; });
    var deduped = [];
    for (var j = 0; j < yPositions.length; j++) {
        if (deduped.length === 0 ||
            Math.abs(yPositions[j] - deduped[deduped.length - 1]) > tolerance) {
            deduped.push(yPositions[j]);
        }
    }

    $.writeln("  Guide Y positions found (mm): [" + deduped.join(", ") + "]");
    return deduped;
}


function computeGuideRatiosForFrame(frameBounds, allGuideYs, tolerance) {
    var frameTop    = frameBounds.top;
    var frameBottom = frameBounds.bottom;
    var frameH      = frameBounds.h;

    var inside = [];
    for (var i = 0; i < allGuideYs.length; i++) {
        var gy = allGuideYs[i];
        // STRICT: guide must be genuinely inside the frame, not just near it
        if (gy > frameTop && gy < frameBottom) {
            inside.push(gy);
        }
    }

    if (inside.length < 2) return null;

    inside.sort(function(a, b) { return a - b; });
    var topGuide    = inside[0];
    var bottomGuide = inside[inside.length - 1];

    return {
        top_ratio:    Math.round(((topGuide    - frameTop) / frameH) * 1000) / 1000,
        bottom_ratio: Math.round(((bottomGuide - frameTop) / frameH) * 1000) / 1000
    };
}


// REPLACE the getGuideYFromLayer function with this:
function getAllGuideYsOnLayer(spread, doc, layerName) {
    /*
     * Collect Y-centre positions of ALL thin lines/rectangles on the
     * named layer, regardless of whether they are grouped or not.
     *
     * v3 FIX: each guide line drawn as a thin rectangle has a top edge
     * AND a bottom edge. We record only the CENTRE of the item (average
     * of top and bottom) so each physical guide produces exactly ONE
     * Y value, not two.  This prevents getGuideRatiosForFrame() from
     * misidentifying the two edges of a single guide as "top guide" and
     * "bottom guide" of the face zone.
     *
     * Returns array of mm centre values sorted top→bottom.
     */
    var yPositions = [];
    var allItems = getAllPageItems(spread);

    for (var i = 0; i < allItems.length; i++) {
        var item = allItems[i];
        if (item.constructor.name !== "Rectangle" &&
            item.constructor.name !== "GraphicLine") continue;
        try {
            if (item.itemLayer.name !== layerName) continue;
        } catch(e) { continue; }

        var b = withMM(doc, function() { return item.geometricBounds; });
        var itemH = Math.abs(b[2] - b[0]);
        var itemW = Math.abs(b[3] - b[1]);
        // Always use the vertical centre of the item as the single Y position
        var centerY = (b[0] + b[2]) / 2;

        // Accept: zero-height lines, thin rectangles up to 10mm tall,
        // or wide near-zero-height rules.
        // The 10mm ceiling catches guides drawn as thick stripes while
        // still excluding actual content frames.
        if (item.constructor.name === "GraphicLine" ||
            itemH < 10 ||
            (itemH < 1.5 && itemW > 5)) {
            yPositions.push(Math.round(centerY * 100) / 100);
        }
    }

    // v4: dedupe EXACT duplicates only (multiple zero-height GraphicLines
    // drawn at the same Y from how the template was authored). The old
    // 8mm merge collapsed distinct guides in staggered-grid layouts where
    // the bottom-of-face-zone of one row sits just 1–8mm above the
    // top-of-face-zone of the next row. Those are different guides for
    // different frames and must be preserved.
    //
    // If a future template draws guides as thick rectangles, getCenterY
    // already gives one value per rectangle, so a 0.5mm tolerance is
    // enough to swallow floating-point rounding without merging real
    // distinct guides.
    yPositions.sort(function(a, b) { return a - b; });
    var deduped = [];
    for (var di = 0; di < yPositions.length; di++) {
        if (deduped.length === 0 ||
            Math.abs(yPositions[di] - deduped[deduped.length - 1]) > 0.5) {
            deduped.push(yPositions[di]);
        }
    }

    $.writeln("  All guide Ys on '" + layerName + "' (centres, deduped): [" + deduped.join(", ") + "]");
    return deduped;
}


// v6: cluster a list of numbers and return the mode (centre + count of the
// largest cluster within `tolerance` mm of its start).
function findClusterMode(values, tolerance) {
    if (values.length === 0) return null;
    values.sort(function(a, b) { return a - b; });
    var bestCount = 0, bestValue = null;
    var s = 0;
    while (s < values.length) {
        var e = s;
        while (e + 1 < values.length && values[e + 1] - values[s] <= tolerance) e++;
        var count = e - s + 1;
        if (count > bestCount) {
            bestCount = count;
            var sum = 0;
            for (var k = s; k <= e; k++) sum += values[k];
            bestValue = sum / count;
        }
        s = e + 1;
    }
    return { value: Math.round(bestValue * 10) / 10, count: bestCount };
}


// v6: detect the template's canonical face-zone POSITION (top & bottom
// offsets from the frame top) by clustering per-frame samples.
//
// Why offsets, not gaps: in a staggered grid, the gap between guides is
// ambiguous because many cross-row pairs share similar gaps. But the
// OFFSET from frame top to the row's top/bottom guide is identical in
// every row (that's the entire point of a uniform layout). So we collect
// "topmost guide offset" once per frame for the top, and "any guide below
// the topmost, expressed as offset from frame top" for the bottom, then
// take the mode of each.
function detectCanonicalOffsets(frames, allGuideYs) {
    var topOffsets = [];
    var bottomOffsets = [];

    for (var fi = 0; fi < frames.length; fi++) {
        var ft = frames[fi].top;
        var fb = frames[fi].bottom;
        var topY = null;
        for (var gi = 0; gi < allGuideYs.length; gi++) {
            var gy = allGuideYs[gi];
            if (gy > ft + 0.5 && gy < fb - 0.5) {
                topY = gy;
                topOffsets.push(gy - ft);
                break; // topmost only
            }
        }
        if (topY === null) continue;
        for (var gj = 0; gj < allGuideYs.length; gj++) {
            var gy2 = allGuideYs[gj];
            // Skip guides within 5mm of the top (those are noise/adjacent
            // duplicates, not the bottom of THIS frame's face zone).
            if (gy2 > topY + 5 && gy2 < fb - 0.5) {
                bottomOffsets.push(gy2 - ft);
            }
        }
    }

    var topR = findClusterMode(topOffsets, 1.5);
    var botR = findClusterMode(bottomOffsets, 1.5);
    if (!topR || !botR) return null;
    return {
        topOffset:    topR.value,
        bottomOffset: botR.value,
        topCount:     topR.count,
        bottomCount:  botR.count
    };
}


// v7: pick the (top, bottom) face-zone pair for the TEACHER frame.
//
// The teacher frame is typically much taller than student frames, so it
// overlaps several student rows whose guides leak inside. Worse, the
// teacher often has its OWN dedicated face-zone guides at different
// relative positions from students (e.g. 5/38 mm in a 63 mm teacher frame
// vs. 3/27.1 mm in a 40 mm student frame), so the student-derived
// canonical points us at the wrong guides.
//
// Strategy: subtract the predictable student-row guide positions
// (`student_top + canonical.topOffset` and `+ canonical.bottomOffset`,
// for every student frame on the spread) from the guides inside the
// teacher frame. Whatever's left is teacher-specific. If we end up with
// ≥2 teacher-specific guides, take the topmost and bottommost — that
// brackets the teacher's own face zone. If we end up with <2 (e.g. a
// template where the teacher reuses student-style guides), fall back to
// the canonical-based picker.
function getTeacherGuideRatios(teacherBounds, allGuideYs, studentFrames, canonical) {
    var STUDENT_MATCH_TOL = 1.5; // mm — guide is "a student-row guide" if within this
    var ft = teacherBounds.top;
    var fb = teacherBounds.bottom;
    var fh = teacherBounds.h;

    // Build the set of predicted student-row guide positions across all
    // student frames. These are the "noise" we want to subtract.
    var studentGuidePositions = [];
    if (canonical && studentFrames && studentFrames.length > 0) {
        for (var i = 0; i < studentFrames.length; i++) {
            studentGuidePositions.push(studentFrames[i].top + canonical.topOffset);
            studentGuidePositions.push(studentFrames[i].top + canonical.bottomOffset);
        }
    }

    var inside = [];
    var teacherSpecific = [];
    for (var g = 0; g < allGuideYs.length; g++) {
        var gy = allGuideYs[g];
        if (gy <= ft + 0.5 || gy >= fb - 0.5) continue;
        inside.push(gy);
        var matchesStudent = false;
        for (var sp = 0; sp < studentGuidePositions.length; sp++) {
            if (Math.abs(gy - studentGuidePositions[sp]) <= STUDENT_MATCH_TOL) {
                matchesStudent = true;
                break;
            }
        }
        if (!matchesStudent) teacherSpecific.push(gy);
    }

    if (teacherSpecific.length >= 2) {
        teacherSpecific.sort(function(a, b) { return a - b; });
        var top = teacherSpecific[0];
        var bot = teacherSpecific[teacherSpecific.length - 1];
        $.writeln("    Teacher frame [" + ft.toFixed(1) + "–" + fb.toFixed(1) + "mm]: " +
                  "inside=[" + inside.join(", ") + "]" +
                  "  teacher-specific=[" + teacherSpecific.join(", ") + "]" +
                  "  → top=" + top + " bot=" + bot);
        return {
            top_ratio:    Math.round(((top - ft) / fh) * 1000) / 1000,
            bottom_ratio: Math.round(((bot - ft) / fh) * 1000) / 1000
        };
    }

    $.writeln("    Teacher frame: only " + teacherSpecific.length +
              " teacher-specific guide(s) found — falling back to canonical match " +
              "(teacher likely reuses student-style face zone)");
    return getGuideRatiosForFrame(teacherBounds, allGuideYs, canonical);
}


// v6: pick the (top, bottom) guide pair for this frame using canonical
// offsets. For each canonical position (frame_top + offset), find the
// closest actual guide inside the frame, within TOL mm. This is robust to
// staggered layouts because the per-row pattern (4mm/31mm in B4, 4mm/27mm
// in A4, etc.) is what defines correctness — not the gap between guides.
function getGuideRatiosForFrame(frameBounds, allGuideYs, canonical) {
    if (!canonical) return null;
    var TOL = 5; // mm tolerance around the canonical position

    var ft = frameBounds.top;
    var fb = frameBounds.bottom;
    var fh = frameBounds.h;
    var targetTopY = ft + canonical.topOffset;
    var targetBotY = ft + canonical.bottomOffset;

    var bestTop = null, bestTopDiff = TOL + 1;
    var bestBot = null, bestBotDiff = TOL + 1;
    var insideForLog = [];

    for (var i = 0; i < allGuideYs.length; i++) {
        var gy = allGuideYs[i];
        if (gy <= ft + 0.5 || gy >= fb - 0.5) continue;
        insideForLog.push(gy);
        var td = Math.abs(gy - targetTopY);
        if (td < bestTopDiff) { bestTop = gy; bestTopDiff = td; }
        var bd = Math.abs(gy - targetBotY);
        if (bd < bestBotDiff) { bestBot = gy; bestBotDiff = bd; }
    }

    if (bestTop === null || bestBot === null || bestTop >= bestBot) {
        $.writeln("    Frame [" + ft.toFixed(1) + "–" + fb.toFixed(1) +
                  "mm]: no valid pair within ±" + TOL + "mm of canonical " +
                  "(top target=" + targetTopY + ", bot target=" + targetBotY + ")");
        return null;
    }

    $.writeln("    Frame [" + ft.toFixed(1) + "–" + fb.toFixed(1) + "mm]: " +
              "inside=[" + insideForLog.join(", ") + "]" +
              "  → top=" + bestTop + " (target " + targetTopY +
              ", Δ" + bestTopDiff.toFixed(1) + ")" +
              "  bot=" + bestBot + " (target " + targetBotY +
              ", Δ" + bestBotDiff.toFixed(1) + ")");
    return {
        top_ratio:    Math.round(((bestTop - ft) / fh) * 1000) / 1000,
        bottom_ratio: Math.round(((bestBot - ft) / fh) * 1000) / 1000
    };
}

// ══════════════════════════════════════════════════════════════════
// SCALE CLAMP CALCULATOR
// Derives a sensible min/max scale clamp from frame aspect ratio.
// Portrait frames that are taller relative to width need higher zoom
// to fill the face zone properly.
// ══════════════════════════════════════════════════════════════════

function deriveScaleClamp(frameW, frameH, guideRatios) {
    // Safety cap only — actual scale is computed per-person by background-based
    // subject-top detection (v12). High max ensures this never binds during
    // normal operation and manual frame_config.json edits are never required.
    return { min: 100, max: 400 };
}


// ══════════════════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════════════════

function zeroPad(n, w) {
    var s = String(n);
    while (s.length < w) s = "0" + s;
    return s;
}

function isoTimestamp() {
    var d = new Date();
    return d.getFullYear() + "-" +
           zeroPad(d.getMonth() + 1, 2) + "-" +
           zeroPad(d.getDate(), 2) + "T" +
           zeroPad(d.getHours(), 2) + ":" +
           zeroPad(d.getMinutes(), 2) + ":" +
           zeroPad(d.getSeconds(), 2);
}

function writeTextFile(path, content) {
    var f = new File(path);
    f.encoding = "UTF-8";
    f.open("w");
    f.write(content);
    f.close();
}

function r(n, dec) {
    var factor = Math.pow(10, dec || 3);
    return Math.round(n * factor) / factor;
}


// ══════════════════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════════════════

(function main() {

    $.writeln("╔══════════════════════════════════════════════╗");
    $.writeln("║  ExportFrameDimensions v2 — Full Measurer    ║");
    $.writeln("╚══════════════════════════════════════════════╝");

    // ── Document ──────────────────────────────────────────────────
    if (app.documents.length === 0) {
        alert("No document is open. Please open your InDesign template first.");
        return;
    }
    var doc = app.activeDocument;
    $.writeln("✓ Document: " + doc.name);

    // ── Spread ────────────────────────────────────────────────────
    var spread;
    if (CONFIG.spreadIndex >= 0) {
        spread = doc.spreads[CONFIG.spreadIndex];
    } else {
        spread = app.layoutWindows[0].activePage.parent;
    }
    $.writeln("✓ Spread: pages " +
        spread.pages[0].name + "–" + spread.pages[spread.pages.length - 1].name);

    // ── Portrait frames ───────────────────────────────────────────
    var allPortraitFrames = getFramesOnLayer(
        spread, CONFIG.portraitLayer, CONFIG.portraitLayerAlt
    );
    $.writeln("  Portrait frames found: " + allPortraitFrames.length);

    if (allPortraitFrames.length < 2) {
        alert("ERROR: Need at least 2 portrait frames.\nFound: " + allPortraitFrames.length +
              "\nCheck CONFIG.portraitLayer (" + CONFIG.portraitLayer + ")");
        return;
    }

    // ── Detect teacher frames ─────────────────────────────────────
    // CRITICAL: this runs BEFORE any "largest = group photo" removal,
    // because in templates where the teacher frame is the largest frame
    // on the spread (and there's no separate group photo), removing the
    // largest first would silently eat the teacher.
    //
    // Strategy 1 (preferred): frames whose object name starts with the
    //   teacherFrameNamePrefix (matches AutoPlace's "T_Photo_*" convention).
    //   Works regardless of size.
    // Strategy 2 (fallback): frames whose size differs from the most-common
    //   size — the original heuristic, kept for templates with no naming
    //   convention but a distinctly-sized teacher.
    // Strategy 3 (last resort): the largest remaining frame.
    var teacherFrameRefs = [];

    // Strategy 1: by name prefix — runs FIRST, before group removal
    var namePrefix = CONFIG.teacherFrameNamePrefix;
    if (namePrefix && namePrefix.length > 0) {
        var notTeacher = [];
        for (var npi = 0; npi < allPortraitFrames.length; npi++) {
            var fname = "";
            try { fname = allPortraitFrames[npi].name || ""; } catch(e) {}
            if (fname.length >= namePrefix.length &&
                fname.substring(0, namePrefix.length) === namePrefix) {
                teacherFrameRefs.push(allPortraitFrames[npi]);
            } else {
                notTeacher.push(allPortraitFrames[npi]);
            }
        }
        if (teacherFrameRefs.length > 0) {
            $.writeln("  Teacher frame(s) found by name prefix '" + namePrefix +
                      "': " + teacherFrameRefs.length);
            allPortraitFrames = notTeacher;
        }
    }

    // ── Identify students (most common size) and remove extras ────
    // With teacher already out of the pool, the student size is the
    // most common size in what remains. Any frame with a different
    // size is an "extra" — group photo, additional decorative frame,
    // OR (if Strategy 1 found no teacher) the teacher itself.
    var sizeCountAll = {};
    for (var pi = 0; pi < allPortraitFrames.length; pi++) {
        var d0 = frameBoundsMM(allPortraitFrames[pi], doc);
        var k0 = r(d0.w, 1) + "x" + r(d0.h, 1);
        sizeCountAll[k0] = (sizeCountAll[k0] || 0) + 1;
    }
    var studentSizeKey = null, studentSizeCount = 0;
    for (var ks in sizeCountAll) {
        if (sizeCountAll.hasOwnProperty(ks) && sizeCountAll[ks] > studentSizeCount) {
            studentSizeCount = sizeCountAll[ks];
            studentSizeKey = ks;
        }
    }
    $.writeln("  Most common frame size (student): " + studentSizeKey +
              " (" + studentSizeCount + " frames)");

    var studentFrames = [];
    var extraFrames  = [];
    for (var pi2 = 0; pi2 < allPortraitFrames.length; pi2++) {
        var d1 = frameBoundsMM(allPortraitFrames[pi2], doc);
        var k1 = r(d1.w, 1) + "x" + r(d1.h, 1);
        if (k1 === studentSizeKey) {
            studentFrames.push(allPortraitFrames[pi2]);
        } else {
            extraFrames.push(allPortraitFrames[pi2]);
        }
    }
    allPortraitFrames = studentFrames;

    // Strategy 2: if no teacher found by name, use the extras
    if (teacherFrameRefs.length === 0 && extraFrames.length > 0) {
        // Of the extras, pick the smallest as teacher; anything bigger is
        // probably the group photo. (Group photos are typically much larger
        // than teacher frames.)
        extraFrames.sort(function(a, b) {
            var da = frameBoundsMM(a, doc);
            var db = frameBoundsMM(b, doc);
            return (da.w * da.h) - (db.w * db.h);
        });
        teacherFrameRefs.push(extraFrames[0]);
        var t0 = frameBoundsMM(extraFrames[0], doc);
        $.writeln("  Teacher frame found by size difference: " +
                  r(t0.w,1) + "×" + r(t0.h,1) + "mm");
        // Log any leftover extras (group photo, etc.)
        for (var ex = 1; ex < extraFrames.length; ex++) {
            var ed = frameBoundsMM(extraFrames[ex], doc);
            $.writeln("    Other non-student-size frame ignored: " +
                      r(ed.w,1) + "×" + r(ed.h,1) + "mm (likely group photo)");
        }
    } else if (extraFrames.length > 0) {
        // Teacher already found by name → extras are pure overflow
        for (var ex2 = 0; ex2 < extraFrames.length; ex2++) {
            var ed2 = frameBoundsMM(extraFrames[ex2], doc);
            $.writeln("    Non-student-size frame ignored: " +
                      r(ed2.w,1) + "×" + r(ed2.h,1) + "mm (likely group photo)");
        }
    }

    // Strategy 3: last resort — largest remaining student frame
    if (teacherFrameRefs.length === 0) {
        var teacherIdx = findLargestFrame(allPortraitFrames);
        if (teacherIdx >= 0) {
            $.writeln("  ⚠ No teacher frame identified by name or by unique size — " +
                      "using largest remaining student frame as teacher " +
                      "(measurements will assume teacher = student dimensions)");
            teacherFrameRefs.push(allPortraitFrames[teacherIdx]);
            allPortraitFrames.splice(teacherIdx, 1);
        }
    }

    $.writeln("  Teacher frame slots found: " + teacherFrameRefs.length);
    // Use the first (leftmost) teacher frame for measurements
    var teacherFrameRef = teacherFrameRefs[0];
    var teacherBounds = frameBoundsMM(teacherFrameRef, doc);
    $.writeln("  Teacher frame (slot 0): " + r(teacherBounds.w,2) + " × " + r(teacherBounds.h,2) + " mm");
    if (teacherFrameRefs.length > 1) {
        $.writeln("  Note: " + teacherFrameRefs.length +
                  " teacher slots detected — only slot 0 measurements exported.");
    }

    // Student frame (most common size)
    var sizeMap = {};
    var sizeFrameMap = {};  // key → first frame with that size (for guide sampling)
    for (var i = 0; i < allPortraitFrames.length; i++) {
        var d = frameBoundsMM(allPortraitFrames[i], doc);
        var key = r(d.w,2) + "x" + r(d.h,2);
        sizeMap[key] = (sizeMap[key] || 0) + 1;
        if (!sizeFrameMap[key]) sizeFrameMap[key] = { frame: allPortraitFrames[i], bounds: d };
    }
    var bestKey = null, bestCount = 0;
    for (var k in sizeMap) {
        if (sizeMap.hasOwnProperty(k) && sizeMap[k] > bestCount) {
            bestCount = sizeMap[k]; bestKey = k;
        }
    }
    var studentW = parseFloat(bestKey.split("x")[0]);
    var studentH = parseFloat(bestKey.split("x")[1]);
    var representativeStudentBounds = sizeFrameMap[bestKey].bounds;
    $.writeln("  Student frame: " + studentW + " × " + studentH +
              " mm  (" + bestCount + " frames)");

    // ── Read guide positions ──────────────────────────────────────
    $.writeln("\n  Reading guide positions on layer '" + CONFIG.guideLayer + "'...");
    var allGuideYs = getAllGuideYsOnLayer(spread, doc, CONFIG.guideLayer);

    // v6: detect canonical face-zone position from the per-frame pattern.
    // The "topmost guide offset from frame top" and "bottom guide offset
    // from frame top" are identical in every row of a sane template; the
    // modes of these per-frame samples give us the template's intended
    // face-zone placement. We feed STUDENT frames to the detector — the
    // teacher frame uses a different face-zone height in many templates,
    // so including it would dilute the mode.
    var studentBoundsList = [];
    for (var sfi = 0; sfi < allPortraitFrames.length; sfi++) {
        studentBoundsList.push(frameBoundsMM(allPortraitFrames[sfi], doc));
    }
    var canonical = detectCanonicalOffsets(studentBoundsList, allGuideYs);
    if (canonical) {
        $.writeln("  Canonical face-zone position (from " + studentBoundsList.length + " student frames):");
        $.writeln("    top    = frame_top + " + canonical.topOffset    + " mm  (" + canonical.topCount    + " agreeing samples)");
        $.writeln("    bottom = frame_top + " + canonical.bottomOffset + " mm  (" + canonical.bottomCount + " agreeing samples)");
    } else {
        $.writeln("  ⚠ Could not detect canonical face-zone position — too few guides");
    }

    // For student guides: sample from the first few student frames and average
    var studentGuideRatios = null;
    var ratioSamples = [];
    for (var si = 0; si < allPortraitFrames.length && si < 5; si++) {
        var sb = frameBoundsMM(allPortraitFrames[si], doc);
        var sr = getGuideRatiosForFrame(sb, allGuideYs, canonical);
        if (sr) {
            ratioSamples.push(sr);
            $.writeln("  Student frame[" + si + "] guide ratios: top=" + sr.top_ratio + " bottom=" + sr.bottom_ratio);
        }
    }
    if (ratioSamples.length > 0) {
        var sumTop = 0, sumBot = 0;
        for (var ri = 0; ri < ratioSamples.length; ri++) {
            sumTop += ratioSamples[ri].top_ratio;
            sumBot += ratioSamples[ri].bottom_ratio;
        }
        studentGuideRatios = {
            top_ratio:    Math.round((sumTop / ratioSamples.length) * 1000) / 1000,
            bottom_ratio: Math.round((sumBot / ratioSamples.length) * 1000) / 1000
        };
        $.writeln("  Student guide ratios (averaged): top=" + studentGuideRatios.top_ratio +
                " bottom=" + studentGuideRatios.bottom_ratio);
    } else {
        $.writeln("  ⚠ No guide lines found inside student frames. Check that lines are on '" + 
                CONFIG.guideLayer + "' layer and their Y positions overlap the frames.");
    }

    // For teacher: use the dedicated teacher picker. It subtracts student-
    // row guide positions from the guides inside the teacher frame, leaving
    // teacher-specific guides — works whether the teacher reuses the student
    // face-zone position or has its own (e.g. a 5/38 mm zone in a 63 mm
    // teacher frame vs. the 3/27 mm zone students use in 40 mm frames).
    var teacherGuideRatios = getTeacherGuideRatios(
        teacherBounds, allGuideYs, studentBoundsList, canonical
    );
    if (teacherGuideRatios) {
        $.writeln("  Teacher guide ratios: top=" + teacherGuideRatios.top_ratio +
                " bottom=" + teacherGuideRatios.bottom_ratio);
    } else {
        $.writeln("  ⚠ No guide lines found inside teacher frame.");
        // Try the student guide ratios as a fallback (rescaled to teacher frame height)
    }

    // ── Derive scale clamps ───────────────────────────────────────
    var studentClamp, teacherClamp;

    if (studentGuideRatios) {
        studentClamp = deriveScaleClamp(studentW, studentH, studentGuideRatios);
        $.writeln("  Student scale clamp: [" + studentClamp.min + "–" + studentClamp.max + "%]");
    } else {
        studentClamp = { min: 125, max: 145 };
        $.writeln("  Student scale clamp: using defaults [125–145%]");
    }

    if (teacherGuideRatios) {
        teacherClamp = deriveScaleClamp(r(teacherBounds.w, 2), r(teacherBounds.h, 2), teacherGuideRatios);
        $.writeln("  Teacher scale clamp: [" + teacherClamp.min + "–" + teacherClamp.max + "%]");
    } else {
        teacherClamp = { min: 120, max: 145 };
        $.writeln("  Teacher scale clamp: using defaults [120–145%]");
    }

    // ── Build JSON ────────────────────────────────────────────────
    var sizeMapStr = "{";
    var first = true;
    for (var sk in sizeMap) {
        if (!sizeMap.hasOwnProperty(sk)) continue;
        if (!first) sizeMapStr += ", ";
        sizeMapStr += '"' + sk + '": ' + sizeMap[sk];
        first = false;
    }
    sizeMapStr += "}";

    function guideRatioStr(gr) {
        if (!gr) return "null";
        return '{ "top_ratio": ' + gr.top_ratio +
               ', "bottom_ratio": ' + gr.bottom_ratio + ' }';
    }
    function clampStr(cl) {
        return '{ "min": ' + cl.min + ', "max": ' + cl.max + ' }';
    }

    var json =
        "{\n" +
        '  "student": {\n' +
        '    "frame_w_mm": '   + studentW           + ',\n' +
        '    "frame_h_mm": '   + studentH           + ',\n' +
        '    "guide_ratios": ' + guideRatioStr(studentGuideRatios) + ',\n' +
        '    "scale_clamp": '  + clampStr(studentClamp) + '\n' +
        '  },\n' +
        '  "teacher": {\n' +
        '    "frame_w_mm": '   + r(teacherBounds.w, 2) + ',\n' +
        '    "frame_h_mm": '   + r(teacherBounds.h, 2) + ',\n' +
        '    "guide_ratios": ' + guideRatioStr(teacherGuideRatios) + ',\n' +
        '    "scale_clamp": '  + clampStr(teacherClamp) + '\n' +
        '  },\n' +
        '  "all_guide_y_positions_mm": [' + allGuideYs.join(", ") + '],\n' +
        '  "all_student_sizes_detected": ' + sizeMapStr + ',\n' +
        '  "source_document": "' + doc.name.replace(/\\/g, "\\\\") + '",\n' +
        '  "generated_at": "' + isoTimestamp() + '"\n' +
        '}';

    writeTextFile(CONFIG.outputFile, json);

    // ── Summary ───────────────────────────────────────────────────
    var guideNote = studentGuideRatios
        ? (" Student Guide ratios: top=" + studentGuideRatios.top_ratio +
           "  chin=" + studentGuideRatios.bottom_ratio)
        : "  ⚠ Student Guide ratios: not found — defaults used\n" +
          "    (Add red hairline rectangles on '" + CONFIG.guideLayer + "' layer)";

        var teacherGuideNote = teacherGuideRatios
                ? (" Teacher Guide ratios: top=" + teacherGuideRatios.top_ratio +
                     "  chin=" + teacherGuideRatios.bottom_ratio)
                : "  ⚠ Teacher Guide ratios: not found — defaults used\n" +
                    "    (Add red hairline rectangles on '" + CONFIG.portraitLayer + "' layer)";


    var summary =
        "\n════════════════════════════════════════════\n" +
        "✓ frame_config.json written\n" +
        "════════════════════════════════════════════\n" +
        "  Student frame : " + studentW + " × " + studentH + " mm\n" +
        "  Teacher frame : " + r(teacherBounds.w,2) + " × " + r(teacherBounds.h,2) + " mm\n" +
        guideNote + "\n" +
                teacherGuideNote + "\n" +
        "  Output        : " + CONFIG.outputFile + "\n\n" +
        "Next step: run face_offset_calculator.py\n" +
        "All target ratios are now computed automatically.\n";

    $.writeln(summary);
    alert(summary);

})();