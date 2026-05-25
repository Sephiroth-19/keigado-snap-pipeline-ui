/*
 * AutoPlacePhotosAndNames v15 — Layer-aware InDesign automation
 * ==============================================================
 *
 * v15 FIX: placeWithTransform now uses ABSOLUTE scale assignment.
 * Fill scales (H/V) are recorded immediately after FILL_PROPORTIONALLY,
 * then final scale is set as fillScale × multiplier (not *=).
 * This makes the scale deterministic regardless of InDesign's internal
 * graphic anchor point, and exactly matches the model in
 * face_offset_calculator.py: graphic centre = frame centre after
 * CENTER_CONTENT, then mm offset applied.
 * =========================================================
 *
 * Reads manifest.json + a package folder from run_pipeline.py and places:
 *   - Student/teacher portrait photos  →  "Default" / "本番カット" layer
 *   - ID plate (札) photos             →  "札持ちカット" layer
 *
 * Name placement is implemented but currently disabled.
 * Search for [NAMES] to find and re-enable all relevant sections.
 *
 * File-naming conventions supported (auto-detected):
 *   NEW  26_千早高_IMG_5678_3101_本01.jpg  /  _札01.jpg
 *   OLD  26_千早高_IMG_5678_3101_本_01.jpg /  _札.jpg
 *
 * Template frame naming:
 *   Portrait  : Photo_NN (students), T_Photo_01 / T_Photo_02 (teachers)
 *   ID plate  : Card_NN  (students), T_Card_01  / T_Card_02  (teachers)
 *   Name label: Label_NN (students), T_Label_01 / T_Label_02 (teachers)  [NAMES]
 *
 * Usage:
 *   1. Open the InDesign template
 *   2. Edit CONFIG below (paths, class ID, offsets)
 *   3. Run via Scripts panel  or  File › Scripts › Run Script
 */

#target indesign

// ══════════════════════════════════════════════════════════════════
// CONFIGURATION — EDIT THESE
// ══════════════════════════════════════════════════════════════════

var CONFIG = {
    // Path to the InDesign template
    indesignFile: "D:\\career\\1_LTID\\Photography\\Client Sample Image\\20260224　個人写真_レイヤー概念など\\26千早高_個人レイアウト_template.indd",

    // Package folder produced by run_pipeline.py
    packageFolder: "D:\\career\\1_LTID\\Photography\\output_千早高_FINAL_v6\\individual_output_19\\3A",

    // manifest.json (one level up from packageFolder)
    manifestFile: "D:\\career\\1_LTID\\Photography\\output_千早高_FINAL_v6\\individual_output_19\\manifest.json",

    // Class ID being placed (letter "A" or grade-class "3-7")
    classLetter: "3A",

    // Group photo (optional — leave empty to skip)
    groupPhotoFile: "",

    // Manual teacher portrait override (leave "" to use manifest entry)
    teacherPhotoFile: "",

    // Output file
    outputFile: "D:\\career\\1_LTID\\Photography\\output_千早高_FINAL_v6\\千早高_Class_3-1_Final.indd",

    // Layer names (must match the template)
    layers: {
        portrait:   "Default",          // also accepts "本番カット"
        idPlate:    "札持ちカット",
        name:       "名前"   // [NAMES] — used when name placement is enabled
    },

    // Absent students list (optional .txt file — numbers separated by comma/newline)
    // Leave empty "" to rely only on file-based absence detection
    absentFile: "",

    // offsetX/Y: shift image inside frame after fitting (mm)
    // scaleFactor: 100 = no change, 110 = 10% larger, etc.
    student: {
        offsetX:     -3.5,
        offsetY:     78.10,
        scaleFactor: 122
    },
    teacher: {
        offsetX:    -2.5,
        offsetY:    10.10,
        scaleFactor: 110
    },
    
    ignoreTeacherManifest: false,
    // ID plate (札) fitting controls
    //   offsetY positive = move down to show full head
    idPlate: {
        offsetX:     -3,
        offsetY:     6.5,
        scaleFactor: 135
    },


    // Behaviour
    autoSave: true,
    autoClose: false,

    // Which spread to place photos on.
    //   -1 = use the currently active/visible spread in InDesign (recommended)
    //    0 = first spread, 1 = second spread, etc.
    spreadIndex: -1
};


// ══════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ══════════════════════════════════════════════════════════════════

function sortFramesByPosition(frames) {
    /*
     * Sort frames into reading order: left-to-right columns,
     * top-to-bottom within each column.
     * Uses a 10mm snap tolerance for column alignment (robust for real templates).
     */
    frames.sort(function (a, b) {
        var ga = a.geometricBounds;   // [top, left, bottom, right]
        var gb = b.geometricBounds;
        var ax = ga[1], ay = ga[0];
        var bx = gb[1], by = gb[0];

        if (Math.abs(ax - bx) > 10) {
            return ax - bx;           // different column → sort by X
        }
        return ay - by;               // same column → sort by Y
    });
    return frames;
}


function findLargestFrame(frames) {
    /* Return index of the largest rectangle (group photo frame). */
    var maxArea = 0;
    var idx = -1;
    for (var i = 0; i < frames.length; i++) {
        var gb = frames[i].geometricBounds;
        var area = (gb[3] - gb[1]) * (gb[2] - gb[0]);
        if (area > maxArea) {
            maxArea = area;
            idx = i;
        }
    }
    return idx;
}



function getFrameName(frame) {
    /*
     * Safely return the frame's label / name as set in the template.
     * InDesign rectangles expose their name via .name or .label.
     * Returns "" if neither is accessible.
     */
    try { if (frame.name  && frame.name.length  > 0) return frame.name;  } catch (e) {}
    try { if (frame.label && frame.label.length > 0) return frame.label; } catch (e) {}
    return "";
}


function findTeacherFramesByName(doc, spread, layerName) {
    /*
     * v21 FIX: Find T_Photo_* frames by scanning the SPREAD's own page items
     * by name — the same reliable approach used for T_Card_* frames.
     *
     * The previous doc.rectangles.itemByName() approach could return a
     * master-page copy of the frame instead of the live document-page override.
     * Calling frame.place() on a master-page item fails silently (the frame
     * object is valid and exists, but the placed image never appears) because
     * InDesign does not allow direct modification of master-page items from a
     * script that targets the document page.
     *
     * By scanning spread.allPageItems first we are guaranteed to receive the
     * actual document-page frame object that can accept placed content.
     *
     * Fallback: doc.rectangles.itemByName() is still tried if the spread scan
     * finds nothing, preserving backward compatibility.
     *
     * Returns { teacherSlots: [frame|null, ...], teacherIds: {id:true, ...} }
     */
    var teacherMap = {};
    var maxSlot    = -1;
    var teacherIds = {};

    // ── Pass 1: scan the spread's portrait-layer frames by name ─────────────
    // This guarantees a live document-page frame that accepts frame.place().
    var spreadItems = getFramesOnLayer(spread, layerName, "本番カット");
    for (var si = 0; si < spreadItems.length; si++) {
        var fn  = getFrameName(spreadItems[si]);
        var fnL = fn.toLowerCase();
        var m   = fnL.match(/^t[_ ]?photo[_ ]?0*(\d+)$/);
        if (!m) continue;
        var slotIdx = parseInt(m[1], 10) - 1;
        if (slotIdx < 0) continue;
        teacherMap[slotIdx] = spreadItems[si];
        teacherIds[spreadItems[si].id] = true;
        if (slotIdx > maxSlot) maxSlot = slotIdx;
        $.writeln("  ✓ Found teacher frame on spread: '" + fn + "' → slot " + slotIdx);
    }

    // ── Pass 2: fallback — doc.rectangles.itemByName() ───────────────────────
    if (maxSlot < 0) {
        $.writeln("  ⚠ No T_Photo_* found on spread — trying doc.rectangles fallback");
        for (var slotNum = 1; slotNum <= 9; slotNum++) {
            if (teacherMap.hasOwnProperty(slotNum - 1)) continue;
            var suffix = (slotNum < 10 ? "0" : "") + slotNum;
            var candidates = [
                "T_Photo_" + suffix, "T_Photo_" + slotNum,
                "T_Photo"  + suffix, "T_Photo"  + slotNum
            ];
            for (var ci = 0; ci < candidates.length; ci++) {
                try {
                    var rect = doc.rectangles.itemByName(candidates[ci]);
                    if (rect && rect.isValid) {
                        var rl = rect.itemLayer ? rect.itemLayer.name : "";
                        if (rl !== layerName && rl !== "本番カット") continue;
                        var slotIdx2 = slotNum - 1;
                        teacherMap[slotIdx2] = rect;
                        teacherIds[rect.id]  = true;
                        if (slotIdx2 > maxSlot) maxSlot = slotIdx2;
                        $.writeln("  ✓ Found teacher frame via doc fallback: '" + candidates[ci] + "' → slot " + slotIdx2);
                        break;
                    }
                } catch (e) {}
            }
        }
    }

    $.writeln("  Teacher slots found: " + (maxSlot + 1));

    // Build dense teacherSlots array
    var teacherSlots = [];
    for (var si2 = 0; si2 <= maxSlot; si2++) {
        teacherSlots.push(teacherMap[si2] || null);
    }

    return { teacherSlots: teacherSlots, teacherIds: teacherIds };
}


function separateTeacherFrames(portraitFrames, teacherIds) {
    /*
     * Split portraitFrames into student pool vs teacher slots,
     * using the teacherIds map produced by findTeacherFramesByName().
     * Falls back to largest-frame heuristic if no teacher IDs were found.
     *
     * v20 FIX: also guards by frame name (T_Photo_*) in case .id comparison
     * fails silently on some InDesign versions — prevents T_Photo_02 from
     * leaking into the student pool and receiving a student portrait.
     */
    var studentFrames = [];

    if (teacherIds && objectHasKeys(teacherIds)) {
        // Name-based split — reliable
        for (var i = 0; i < portraitFrames.length; i++) {
            var isTeacher = false;
            // Primary check: by frame object id
            try { if (teacherIds[portraitFrames[i].id]) isTeacher = true; } catch (e) {}
            // Secondary check: by frame name (catches id-mismatch edge cases)
            if (!isTeacher) {
                var fn = getFrameName(portraitFrames[i]).toLowerCase();
                if (/^t[_ ]?photo[_ ]?\d+$/.test(fn)) {
                    isTeacher = true;
                    $.writeln("  ⚠ Frame '" + getFrameName(portraitFrames[i]) +
                              "' excluded from student pool by name (id lookup missed it)");
                }
            }
            if (!isTeacher) studentFrames.push(portraitFrames[i]);
        }
    } else {
        // Fallback: remove the largest frame (old heuristic)
        $.writeln("  ⚠ No T_Photo_* frames found by name — falling back to largest-frame heuristic");
        var largestIdx = findLargestFrame(portraitFrames);
        for (var j = 0; j < portraitFrames.length; j++) {
            if (j !== largestIdx) studentFrames.push(portraitFrames[j]);
        }
    }

    return studentFrames;
}


function objectHasKeys(obj) {
    for (var k in obj) { if (obj.hasOwnProperty(k)) return true; }
    return false;
}


function normalizePath(p) {
    /*
     * Convert Windows-style backslash paths to forward slashes so that
     * ExtendScript's File() constructor always resolves them correctly,
     * even when packageFolder was typed with backslashes in CONFIG.
     * Also strips a trailing slash so we can always append "/" + filename.
     */
    if (!p) return p;
    p = p.replace(/\\/g, "/");
    if (p.charAt(p.length - 1) === "/") p = p.substring(0, p.length - 1);
    return p;
}


function resolveTeacherFile(fileName, packageFolder) {
    /*
     * Try to locate a teacher photo file.
     * Search order:
     *   1. packageFolder  (already the class sub-folder, e.g. .../3-1)
     *   2. parent of packageFolder  (root output folder)
     *   3. parent + classId sub-folder (for when packageFolder is the root)
     * Returns a File object that exists, or the last attempted File object
     * (for error reporting) when not found.
     */
    var base = normalizePath(packageFolder);
    var tries = [
        new File(base + "/" + fileName),
        new File(new Folder(base).parent ? normalizePath(new Folder(base).parent.fsName) + "/" + fileName : base + "/" + fileName)
    ];
    for (var i = 0; i < tries.length; i++) {
        if (tries[i].exists) return tries[i];
        $.writeln("    [resolveTeacherFile] not found: " + tries[i].fsName);
    }
    return tries[tries.length - 1];   // return last attempt for error message
}


function smartFit(frame, scaleFactor, offsetX, offsetY) {
    // ID-plate (札) frames. Same unit-safe approach as placeWithTransform.
    try {
        frame.fit(FitOptions.FILL_PROPORTIONALLY);
        frame.fit(FitOptions.CENTER_CONTENT);

        if (frame.allGraphics.length > 0) {
            var g = frame.allGraphics[0];

            if (scaleFactor && scaleFactor !== 100) {
                var s = scaleFactor / 100;
                g.horizontalScale *= s;
                g.verticalScale   *= s;
            }

            var ox = (offsetX !== undefined) ? offsetX : 0;
            var oy = (offsetY !== undefined) ? offsetY : 0;
            if (ox !== 0 || oy !== 0) {
                var doc        = frame.parentPage.parent.parent;
                var origHUnits = doc.viewPreferences.horizontalMeasurementUnits;
                var origVUnits = doc.viewPreferences.verticalMeasurementUnits;
                doc.viewPreferences.horizontalMeasurementUnits = MeasurementUnits.MILLIMETERS;
                doc.viewPreferences.verticalMeasurementUnits   = MeasurementUnits.MILLIMETERS;

                g.move(undefined, [ox, oy]);

                doc.viewPreferences.horizontalMeasurementUnits = origHUnits;
                doc.viewPreferences.verticalMeasurementUnits   = origVUnits;
            } else {
                frame.fit(FitOptions.CENTER_CONTENT);
            }
        }
    } catch (e) {
        $.writeln("smartFit error: " + e.message);
    }
}


function placeWithTransform(frame, imageFile, offsetX, offsetY, scaleFactor) {
    /*
     * place → FILL_PROPORTIONALLY → record fill scales → scale absolutely → offset.
     *
     * v15 FIX — eliminates the double CENTER_CONTENT / re-centre ambiguity:
     *
     * The v14 approach (FILL → CENTER → scale *= sp → CENTER again → move) had a
     * subtle error: after "graphic.horizontalScale *= sp", InDesign scales from the
     * graphic object's current top-left corner (which is outside the frame when the
     * image overflows). The second CENTER_CONTENT then repositions the graphic, but
     * the resulting graphic position is NOT identical to what face_offset_calculator.py
     * models as "graphic centre at frame centre".  In practice the graphic drifts
     * slightly, making the computed offsetY incorrect.
     *
     * v15 CONTRACT with face_offset_calculator.py v9+:
     *   scaleFactor  = absolute final scale AS A MULTIPLIER ON FILL base.
     *                  e.g. 145 means the final image is 1.45× the FILL size.
     *   offsetY (mm) = downward shift applied AFTER absolute-scale + re-centre.
     *                  Python models the graphic centre at frame centre after this.
     *   offsetX (mm) = horizontal shift (0 = keep centred).
     *
     * IMPLEMENTATION:
     *   1. FILL_PROPORTIONALLY sets hScale/vScale so image just fills the frame.
     *      Record these "fill scales" (fillH, fillV) immediately.
     *   2. Set absolute final scale:  hScale = fillH * sp,  vScale = fillV * sp.
     *      Because we are SETTING (not multiplying), the result is deterministic
     *      and independent of InDesign's internal pivot behaviour.
     *   3. CENTER_CONTENT re-centres the scaled graphic so its centre = frame centre.
     *      This matches exactly the pivot model in compute_offsets() in Python.
     *   4. graphic.move([ox, oy]) shifts in mm (rulers forced to mm).
     *
     * UNIT SAFETY: rulers forced to mm around the move call.
     */
    try {
        frame.place(imageFile);

        // ── Step 1: FILL — record the fill scales immediately ──────────────
        frame.fit(FitOptions.FILL_PROPORTIONALLY);

        if (frame.allGraphics.length === 0) {
            $.writeln("  ✗ placeWithTransform: no graphic after place()");
            return;
        }
        var graphic = frame.allGraphics[0];
        var fillH   = graphic.horizontalScale;   // e.g. 83.33 (%)
        var fillV   = graphic.verticalScale;

        $.writeln("  [v15] fill scales: H=" + fillH + "%  V=" + fillV + "%");

        // ── Step 2: Apply absolute final scale ─────────────────────────────
        // scaleFactor is "% of FILL size", e.g. 145 = 1.45× the FILL result.
        var sp = (scaleFactor && scaleFactor !== 100) ? (scaleFactor / 100) : 1.0;
        graphic.horizontalScale = fillH * sp;
        graphic.verticalScale   = fillV * sp;

        // ── Step 3: Re-centre — puts graphic centre at frame centre ────────
        // This is the exact pivot assumed by compute_offsets() in Python.
        frame.fit(FitOptions.CENTER_CONTENT);

        // ── Step 4: Apply face-alignment offset ────────────────────────────
        var ox = offsetX || 0;
        var oy = offsetY || 0;
        if (ox !== 0 || oy !== 0) {
            var doc        = frame.parentPage.parent.parent;
            var origHUnits = doc.viewPreferences.horizontalMeasurementUnits;
            var origVUnits = doc.viewPreferences.verticalMeasurementUnits;
            doc.viewPreferences.horizontalMeasurementUnits = MeasurementUnits.MILLIMETERS;
            doc.viewPreferences.verticalMeasurementUnits   = MeasurementUnits.MILLIMETERS;

            graphic.move(undefined, [ox, oy]);

            // ── Debug: verify actual graphic centre Y in frame (remove when confirmed) ──
            var gb = graphic.geometricBounds;   // [top, left, bottom, right] page coords
            var fb = frame.geometricBounds;
            var actualCentreY_inFrame   = ((gb[0] + gb[2]) / 2) - fb[0];
            var expectedCentreY_inFrame = (fb[2] - fb[0]) / 2 + oy;
            $.writeln("  [DBG] gfx-centre in frame: actual=" +
                      actualCentreY_inFrame.toFixed(2) + "mm  expected=" +
                      expectedCentreY_inFrame.toFixed(2) + "mm");

            doc.viewPreferences.horizontalMeasurementUnits = origHUnits;
            doc.viewPreferences.verticalMeasurementUnits   = origVUnits;
        }

        $.writeln("  ✓ [v15] fillH=" + fillH + "% × " + sp +
                  " → final=" + Math.round(fillH * sp) + "%" +
                  "  re-centred, offset(" + ox + "mm, " + oy + "mm)");

    } catch (e) {
        $.writeln("  ✗ placeWithTransform error: " + e.message);
    }
}


function toNumberOrDefault(value, fallback) {
    var n = parseFloat(value);
    return isNaN(n) ? fallback : n;
}


function resolveFacePlacement(entry, fallbackSettings) {
    /*
     * Use per-person manifest face_offsets when available,
     * otherwise fall back to CONFIG defaults.
     */
    var fo = (entry && entry.face_offsets) ? entry.face_offsets : null;
    return {
        offsetX: toNumberOrDefault(fo ? fo.offsetX : undefined, fallbackSettings.offsetX),
        offsetY: toNumberOrDefault(fo ? fo.offsetY : undefined, fallbackSettings.offsetY),
        scaleFactor: toNumberOrDefault(fo ? fo.scaleFactor : undefined, fallbackSettings.scaleFactor),
        source: fo ? "manifest.face_offsets" : "CONFIG fallback"
    };
}


function readAbsentNumbers(filePath) {
    /*
     * Teammate update: read a plain-text absent.txt and return an array
     * of absent student numbers.  Numbers can be comma- or newline-separated.
     */
    var list = [];
    if (!filePath || filePath.length === 0) return list;
    var f = new File(filePath);
    if (!f.exists) { $.writeln("⚠ absent.txt not found — skipping file-based absent list"); return list; }
    f.open("r");
    var content = f.read();
    f.close();
    content = content.replace(/\r\n/g, ",").replace(/\n/g, ",").replace(/\s+/g, ",");
    var parts = content.split(",");
    for (var i = 0; i < parts.length; i++) {
        var n = parseInt(parts[i], 10);
        if (!isNaN(n) && n > 0) list.push(n);
    }
    $.writeln("Absent (from file): " + list.join(", "));
    return list;
}


function isAbsent(num, list) {
    for (var i = 0; i < list.length; i++) {
        if (list[i] === num) return true;
    }
    return false;
}


function readJSON(filePath) {
    var f = new File(filePath);
    if (!f.exists) {
        $.writeln("✗ JSON file not found: " + filePath);
        return null;
    }
    f.open("r");
    var raw = f.read();
    f.close();
    // ExtendScript doesn't have JSON.parse — use eval (safe for our own files)
    try {
        return eval("(" + raw + ")");
    } catch (e) {
        $.writeln("✗ JSON parse error: " + e.message);
        return null;
    }
}


function getFramesOnLayer(spread, layerName, altLayerName) {
    /*
     * Returns all graphic-frame containers on the given layer.
     * Skips child Image/EPS objects and TextFrames so each frame
     * is counted exactly once, even when it already contains placed content.
     */
    var frames = [];
    var allItems = spread.allPageItems;

    // Types that are definitely NOT photo containers
    var SKIP = { "TextFrame":1, "Group":1, "Oval":1, "Polygon":1,
                 "GraphicLine":1, "Image":1, "EPS":1, "PDF":1,
                 "WMF":1, "PICT":1, "ImportedPage":1 };

    for (var i = 0; i < allItems.length; i++) {
        var item = allItems[i];
        if (SKIP[item.constructor.name]) continue;
        try {
            var lName = item.itemLayer.name;
            if (lName === layerName || (altLayerName && lName === altLayerName)) {
                frames.push(item);
            }
        } catch (e) {}
    }
    return frames;
}


function getTextFramesOnLayer(spread, layerName) {
    var frames = [];
    var allItems = spread.allPageItems;
    for (var i = 0; i < allItems.length; i++) {
        var item = allItems[i];
        if (item.constructor.name !== "TextFrame") continue;
        try {
            if (item.itemLayer.name === layerName) {
                frames.push(item);
            }
        } catch (e) {}
    }
    return frames;
}


// function formatNameForTemplate(name) {
//     /*
//      * Replicates the InDesign template's kanji-spacing convention:
//      *   - A fullwidth underscore \uff3f (＿) is inserted between every pair
//      *     of consecutive kanji characters (CJK ideographs).
//      *   - A space (half-width or full-width) that sits between kanji on both
//      *     sides is also replaced by ＿ (so "臼田 安那" → "臼＿田＿安＿那").
//      *   - Non-kanji characters (katakana, hiragana, latin, digits) and the
//      *     spaces around them are left untouched.
//      *
//      * Examples:
//      *   "臼田 安那"          → "臼＿田＿安＿那"
//      *   "梅澤 愛蘭"          → "梅＿澤＿愛＿蘭"
//      *   "アーカー ポンミャッ ライン" → unchanged
//      *   "一ノ瀬 晴"          → "一ノ瀬＿晴"  (ノ breaks the kanji run)
//      */
//     if (!name) return name;

//     var U = "\uff3f"; // ＿ fullwidth underscore

//     function isKanji(ch) {
//         if (!ch) return false;
//         var c = ch.charCodeAt(0);
//         // CJK Unified Ideographs U+4E00–U+9FFF
//         // CJK Extension A       U+3400–U+4DBF
//         // CJK Compat Ideographs  U+F900–U+FAFF
//         return (c >= 0x4E00 && c <= 0x9FFF) ||
//                (c >= 0x3400 && c <= 0x4DBF) ||
//                (c >= 0xF900 && c <= 0xFAFF);
//     }

//     // Pass 1: replace spaces that sit between kanji on both sides with ＿
//     var p1 = "";
//     for (var i = 0; i < name.length; i++) {
//         var ch = name.charAt(i);
//         if (ch === " " || ch === "\u3000") {
//             var leftK = false, rightK = false;
//             for (var l = i - 1; l >= 0; l--) {
//                 if (name.charAt(l) !== " " && name.charAt(l) !== "\u3000") {
//                     leftK = isKanji(name.charAt(l)); break;
//                 }
//             }
//             for (var r = i + 1; r < name.length; r++) {
//                 if (name.charAt(r) !== " " && name.charAt(r) !== "\u3000") {
//                     rightK = isKanji(name.charAt(r)); break;
//                 }
//             }
//             p1 += (leftK && rightK) ? U : ch;
//         } else {
//             p1 += ch;
//         }
//     }

//     // Pass 2: insert ＿ between every pair of directly adjacent kanji
//     var res = "";
//     for (var j = 0; j < p1.length; j++) {
//         res += p1.charAt(j);
//         if (isKanji(p1.charAt(j)) && isKanji(p1.charAt(j + 1))) {
//             res += U;
//         }
//     }
//     return res;
// }


// function sampleUnderscoreStyle(story) {
//     /*
//      * Before we overwrite the story content, scan the existing text for a
//      * ＿ character (U+FF3F) and capture its character style + fill color.
//      * Returns an object { charStyle, fillColor } or null if none found.
//      * We use this to re-apply the template's cyan underscore styling after
//      * writing new names.
//      */
//     try {
//         var chars = story.characters;
//         for (var i = 0; i < chars.length; i++) {
//             if (chars[i].contents === "\uff3f") {
//                 var cs = null, fc = null;
//                 try { cs = chars[i].appliedCharacterStyle; } catch (e) {}
//                 try { fc = chars[i].fillColor; } catch (e) {}
//                 if (cs || fc) {
//                     $.writeln("  ✓ Sampled underscore style from template");
//                     return { charStyle: cs, fillColor: fc };
//                 }
//             }
//         }
//     } catch (e) {}
//     return null;
// }


// function applyUnderscoreStyle(story, styleInfo) {
//     /*
//      * After writing new name text, find every ＿ (U+FF3F) character in the
//      * story and re-apply the sampled character style and fill color so the
//      * underscores look identical to the original template (cyan, styled).
//      */
//     if (!styleInfo) return;
//     try {
//         var chars = story.characters;
//         var count = 0;
//         for (var i = 0; i < chars.length; i++) {
//             if (chars[i].contents === "\uff3f") {
//                 try {
//                     if (styleInfo.charStyle && styleInfo.charStyle.isValid) {
//                         chars[i].appliedCharacterStyle = styleInfo.charStyle;
//                     }
//                     if (styleInfo.fillColor && styleInfo.fillColor.isValid) {
//                         chars[i].fillColor = styleInfo.fillColor;
//                     }
//                     count++;
//                 } catch (e) {}
//             }
//         }
//         if (count > 0) $.writeln("  ✓ Underscore style applied to " + count + " characters");
//     } catch (e) {
//         $.writeln("  ⚠ applyUnderscoreStyle error: " + e.message);
//     }
// }


// function centerNameFrameText(frame) {
//     /*
//      * Center text inside a name frame both horizontally and vertically.
//      * Works for standalone and threaded text frames.
//      */
//     if (!frame) return;
//     try {
//         frame.textFramePreferences.verticalJustification = VerticalJustification.CENTER_ALIGN;
//     } catch (e) {}

//     try {
//         frame.texts[0].justification = Justification.CENTER_ALIGN;
//     } catch (e2) {}
// }


// function placeNamesInFrames(nameFrames, frameNames) {
//     /*
//      * Correctly places one name per text frame, handling both threaded
//      * and non-threaded (standalone) frames.
//      *
//      * frameNames: array indexed by frame position (same order as nameFrames)
//      *             each entry is a name string, or "" for absent/no name.
//      *
//      * Also preserves the template's ＿ character styling (e.g. cyan color)
//      * by sampling it from the existing template content before overwriting,
//      * then re-applying it to every ＿ in the newly written text.
//      */

//     // ── Group frames by their parentStory ──
//     var groups = [];   // [{storyRef, frames:[{frame,idx}]}]

//     for (var i = 0; i < nameFrames.length; i++) {
//         var tf = nameFrames[i];
//         var storyRef;
//         try { storyRef = tf.parentStory; } catch (e) { storyRef = null; }

//         var found = -1;
//         for (var g = 0; g < groups.length; g++) {
//             if (groups[g].storyRef === storyRef) { found = g; break; }
//         }
//         if (found >= 0) {
//             groups[found].frames.push({ frame: tf, idx: i });
//         } else {
//             groups.push({ storyRef: storyRef, frames: [{ frame: tf, idx: i }] });
//         }
//     }

//     $.writeln("  Name story groups: " + groups.length +
//               " (" + nameFrames.length + " frames total)");

//     // ── Fill each group ──
//     for (var g = 0; g < groups.length; g++) {
//         var grp = groups[g];
//         // Ensure frames are in position order within the group
//         grp.frames.sort(function (a, b) { return a.idx - b.idx; });

//         if (grp.frames.length === 1) {
//             // ── Standalone frame: clear then set ──
//             var tf = grp.frames[0].frame;
//             var name = frameNames[grp.frames[0].idx] || "";
//             try { tf.contents = ""; } catch (e) {}
//             if (name) {
//                 try {
//                     tf.contents = name;
//                     $.writeln("  ✓ frame[" + grp.frames[0].idx + "] = " + name);
//                     // Auto-shrink if name overflows the standalone frame
//                     var minPt = 6;
//                     var shrinkAttempts = 0;
//                     while (tf.overflows && shrinkAttempts < 40) {
//                         try {
//                             var chars = tf.parentStory.characters;
//                             var curSz = chars[0].pointSize;
//                             if (curSz <= minPt) break;
//                             for (var sci = 0; sci < chars.length; sci++) {
//                                 chars[sci].pointSize = curSz - 0.5;
//                             }
//                         } catch (e2) { break; }
//                         shrinkAttempts++;
//                     }
//                     if (shrinkAttempts > 0) $.writeln("  ⚠ Shrunk standalone frame font ×" + shrinkAttempts + " for: " + name);
//                 } catch (e) {
//                     $.writeln("  ✗ frame[" + grp.frames[0].idx + "] error: " + e.message);
//                 }
//             }

//             // Keep name centered in the frame regardless of content length.
//             centerNameFrameText(tf);
//         } else {
//             // ── Threaded story: set whole story at once ──
//             var parts = [];
//             for (var j = 0; j < grp.frames.length; j++) {
//                 parts.push(frameNames[grp.frames[j].idx] || "");
//             }
//             var fullContent = parts.join("\r");
//             try {
//                 grp.storyRef.contents = fullContent;
//                 $.writeln("  ✓ threaded story (" + grp.frames.length +
//                           " frames) filled with " + grp.frames.length + " names");
//             } catch (e) {
//                 $.writeln("  ⚠ story.contents failed (" + e.message +
//                           "), trying frame[0].contents fallback");
//                 try { grp.frames[0].frame.contents = fullContent; } catch (e2) {}
//             }

//             // ── Auto-shrink: ensure each paragraph stays within its assigned frame ──
//             // If a long name overflows its frame it pushes all subsequent names
//             // one frame forward. Fix: reduce that paragraph's font size until the
//             // next paragraph's first character lands in the expected next frame.
//             var minPtSize = 6;
//             var storyParas = grp.storyRef.paragraphs;
//             for (var pi = 0; pi < storyParas.length - 1 && pi < grp.frames.length - 1; pi++) {
//                 var thisPara   = storyParas[pi];
//                 var nextPara   = storyParas[pi + 1];
//                 var nextFrame  = grp.frames[pi + 1].frame;
//                 for (var att = 0; att < 60; att++) {
//                     try {
//                         // Check where the first character of the NEXT paragraph is displayed
//                         if (nextPara.characters.length === 0) break;
//                         var nextFirstChar = nextPara.characters[0];
//                         var dispFrames = nextFirstChar.parentTextFrames;
//                         var inRightFrame = false;
//                         for (var dfi = 0; dfi < dispFrames.length; dfi++) {
//                             if (dispFrames[dfi] === nextFrame) { inRightFrame = true; break; }
//                         }
//                         if (inRightFrame) break; // this para fits correctly

//                         // This paragraph overflows — shrink its font by 0.5pt
//                         var paraChars = thisPara.characters;
//                         if (paraChars.length === 0) break;
//                         var curSz2 = paraChars[0].pointSize;
//                         if (curSz2 <= minPtSize) break;
//                         var newSz = curSz2 - 0.5;
//                         for (var pci = 0; pci < paraChars.length; pci++) {
//                             try { paraChars[pci].pointSize = newSz; } catch (e3) {}
//                         }
//                     } catch (e4) { break; }
//                 }
//                 if (att > 0) $.writeln("  ⚠ Shrunk para[" + pi + "] font ×" + att + " to fit frame");
//             }

//             // Center text in every frame in the threaded chain.
//             for (var cj = 0; cj < grp.frames.length; cj++) {
//                 centerNameFrameText(grp.frames[cj].frame);
//             }
//         }
//     }
// }

function buildFileMap(folderPath) {
    /*
     * Scan the package folder and build a lookup:
    *   { "1": { "本01": File, "札01": File }, "2": { ... }, ... }
     *
     * Supports BOTH naming conventions with automatic fallback:
     * 
     * NEW convention: [Year]_[SchoolName]_[OriginalFileName]_[Grade][Class][ID]_[Tag].ext
     *   Example: 26_千早高_IMG_5678_3101_本01.jpg  → number=1, portrait priority=1
     *            26_千早高_IMG_1234_3101_札02.jpg → number=1, ID-plate priority=2
     *
     * OLD convention (v10): [Year]_[SchoolName]_[OriginalFileName]_[Grade][Class][ID]_[Tag].ext
     *   Example: 26_千早高_IMG_5678_3101_本_01.jpg → number=1, portrait priority=1
     *            26_千早高_IMG_1234_3101_札.jpg    → number=1, ID-plate
     *
     * We keep only files the script needs per student:
     *   - Portrait: prefer 本01, fallback 本02, 本03, ...
     *   - ID plate: prefer 札01, fallback 札02, 札03, ...
     *
    * Internal map keys (normalized):
    *   "本01" = selected portrait file
    *   "札01" = selected ID-plate file
     *
     * Teacher files such as "..._31先生_札.JPG" are ignored here
     * because teacher placement is driven by manifest entries.
     */

    function parseStudentFileName(fname) {
        var base = fname.replace(/\.[^.]+$/, "");
        var parts = base.split("_");
        var idToken = null;
        var kind = null;
        var priority = null;

        // Try NEW convention first: ..._[GradeClassID]_札01 or ..._[GradeClassID]_本01
        if (parts.length >= 2) {
            var last = parts[parts.length - 1];
            var mId = last.match(/^札(\d+)$/);
            var mPortrait = last.match(/^本(\d+)$/);

            if (mId) {
                idToken = parts[parts.length - 2];
                kind = "札01";
                priority = parseInt(mId[1], 10);
            } else if (mPortrait) {
                idToken = parts[parts.length - 2];
                kind = "本01";
                priority = parseInt(mPortrait[1], 10);
            }
        }

        // Try OLD convention (v10): ..._[GradeClassID]_札 or ..._[GradeClassID]_本_01
        if (!kind && parts.length >= 2) {
            var last = parts[parts.length - 1];
            
            // Old style ID plate: ..._[GradeClassID]_札
            if (last === "札") {
                idToken = parts[parts.length - 2];
                kind = "札01";
                priority = 1; // Old style札 has no priority number, treat as priority 1
            }
            // Old style portrait: ..._[GradeClassID]_本_01
            else if (parts.length >= 3 && parts[parts.length - 2] === "本" && /^\d+$/.test(last)) {
                idToken = parts[parts.length - 3];
                kind = "本01";
                priority = parseInt(last, 10);
            }
        }

        if (idToken && kind) {
            // Teacher token example: 31先生
            if (idToken.indexOf("先生") !== -1) return null;

            // [Grade][Class][ID] → use trailing 2 digits as student ID
            // Examples: 3101 -> 1, 3312 -> 12, 31001 -> 1
            var tail2 = idToken.match(/(\d{2})$/);
            var num = null;
            if (tail2) {
                num = parseInt(tail2[1], 10);
            } else if (/^\d+$/.test(idToken)) {
                // Fallback for older/simple numeric token
                num = parseInt(idToken, 10);
            }

            if (!isNaN(num) && num > 0) {
                return { number: num, kind: kind, priority: priority || 9999 };
            }
            return null;
        }

        return null;
    }

    var folder = new Folder(folderPath);
    if (!folder.exists) {
        $.writeln("✗ Package folder not found: " + folderPath);
        return {};
    }

    var files = folder.getFiles(/\.(jpg|jpeg|png|tif|tiff|psd)$/i);
    var map = {};

    for (var i = 0; i < files.length; i++) {
        var fname = decodeURI(files[i].name);
        var parsed = parseStudentFileName(fname);
        if (!parsed) continue;

        var num = parsed.number;
        var kind = parsed.kind;
        var priority = parsed.priority;
        var key = String(num);

        if (!map[key]) map[key] = {};

        // Keep the lowest-numbered shot for each required kind.
        // (e.g. prefer 本01 over 本02; 札01 over 札02)
        if (!map[key][kind] || priority < map[key][kind].priority) {
            map[key][kind] = { file: files[i], priority: priority };
        }
    }

    // Flatten wrappers so placement logic gets File objects directly.
    for (var skey in map) {
        if (!map.hasOwnProperty(skey)) continue;
        for (var tkey in map[skey]) {
            if (!map[skey].hasOwnProperty(tkey)) continue;
            map[skey][tkey] = map[skey][tkey].file;
        }
    }

    return map;
}

// Updated highlightAbsentFrame to place a PSD file instead of coloring
function highlightAbsentFrame(frame) {
    try {
        var absentImagePath = "D:\\career\\LTID\\Photography\\Client Sample Image\\20260306　篠崎高_個人_39人ブランク\\absent-bg.png";
        var absentImage = new File(absentImagePath);

        if (!absentImage.exists) {
            $.writeln("⚠ Absent image file not found: " + absentImagePath);
            return;
        }

        frame.place(absentImage);
        frame.fit(FitOptions.FILL_PROPORTIONALLY);
        frame.fit(FitOptions.CENTER_CONTENT);

        $.writeln("  ✓ Absent frame filled with image: " + absentImagePath);
    } catch (e) {
        $.writeln("  ✗ Error placing absent image: " + e.message);
    }
}


// ══════════════════════════════════════════════════════════════════
// MAIN SCRIPT
// ══════════════════════════════════════════════════════════════════

(function main() {

    $.writeln("╔══════════════════════════════╗");
    $.writeln("║  AutoPlacePhotos  v24        ║");
    $.writeln("╚══════════════════════════════╝");

    // ── Validate paths ──
    var templateFile = new File(CONFIG.indesignFile);
    if (!templateFile.exists) {
        alert("ERROR: InDesign template not found!\n" + CONFIG.indesignFile);
        return;
    }

    var pkgFolder = new Folder(CONFIG.packageFolder);
    if (!pkgFolder.exists) {
        alert("ERROR: Package folder not found!\n" + CONFIG.packageFolder);
        return;
    }

    // ── Read manifest ──
    var manifest = null;
    var nameMap = {};    // number -> name string
    var entryMap = {};   // number -> full manifest entry

    if (CONFIG.manifestFile) {
        var fullManifest = readJSON(CONFIG.manifestFile);
        if (fullManifest && fullManifest.classes && fullManifest.classes[CONFIG.classLetter]) {
            manifest = fullManifest.classes[CONFIG.classLetter];
            // Build name lookup
            var entries = manifest.entries || [];
            for (var e = 0; e < entries.length; e++) {
                nameMap[String(entries[e].number)] = entries[e].name;
                entryMap[String(entries[e].number)] = entries[e];
            }
            $.writeln("✓ Manifest loaded: " + entries.length + " entries");
        } else {
            $.writeln("⚠ Manifest not found or class missing — will skip name injection");
        }
    }

    // ── Build file map ──
    var fileMap = buildFileMap(CONFIG.packageFolder);
    var studentNums = [];
    for (var k in fileMap) {
        if (fileMap.hasOwnProperty(k)) studentNums.push(parseInt(k, 10));
    }
    studentNums.sort(function(a, b) { return a - b; });
    $.writeln("✓ Found files for students: " + studentNums.join(", "));

    // ── Open document (or use already-open one) ──
    var doc;

    if (app.documents.length > 0) {
        doc = app.activeDocument;
        $.writeln("✓ Using active document: " + doc.name);
    } else {
        doc = app.open(templateFile);
        $.writeln("✓ Opened template: " + doc.name);
    }

    // ── Select the correct spread ──
    var spread;
    if (CONFIG.spreadIndex >= 0) {
        if (CONFIG.spreadIndex >= doc.spreads.length) {
            alert("ERROR: spreadIndex " + CONFIG.spreadIndex + " is out of range. Document has " + doc.spreads.length + " spreads (0–" + (doc.spreads.length - 1) + ").");
            return;
        }
        spread = doc.spreads[CONFIG.spreadIndex];
        $.writeln("✓ Using spread index " + CONFIG.spreadIndex + " (pages " + spread.pages[0].name + "–" + spread.pages[spread.pages.length - 1].name + ")");
    } else {
        // Use spread of currently active page (safer than activeSpread)
        if (app.layoutWindows.length === 0) {
            alert("No active layout window found.");
            return;
        }

        var activePage = app.layoutWindows[0].activePage;
        spread = activePage.parent;

        $.writeln("✓ Using spread of active page: " +
            spread.pages[0].name + "–" +
            spread.pages[spread.pages.length - 1].name);
    }

    // ── Get portrait frames (Default / 本番カット layer) ──
    var portraitFrames = getFramesOnLayer(spread, CONFIG.layers.portrait, "本番カット");
    $.writeln("  Portrait frames on '" + CONFIG.layers.portrait + "': " + portraitFrames.length);

    // ── Get ID plate frames (札持ちカット layer) ──
    var idPlateFrames = getFramesOnLayer(spread, CONFIG.layers.idPlate);
    $.writeln("  ID plate frames on '" + CONFIG.layers.idPlate + "': " + idPlateFrames.length);

    // [NAMES] ── Get name text frames ──
    // var nameFrames = getTextFramesOnLayer(spread, CONFIG.layers.name);
    // $.writeln("  Name text frames on '" + CONFIG.layers.name + "': " + nameFrames.length);
    var nameFrames = []; // [NAMES] placeholder — re-enable above lines when needed

    // ── Separate group frame from portrait frames ──
    var groupFrameIdx = findLargestFrame(portraitFrames);
    var groupFrame = null;
    if (groupFrameIdx >= 0) {
        groupFrame = portraitFrames[groupFrameIdx];
        portraitFrames.splice(groupFrameIdx, 1);
        $.writeln("✓ Group photo frame identified");
    }


    // ── Find teacher portrait frames by name (T_Photo_01, T_Photo_02, …) ──
    var tPhotoResult  = findTeacherFramesByName(doc, spread, CONFIG.layers.portrait);
    var teacherSlots  = tPhotoResult.teacherSlots;
    var teacherIdMap  = tPhotoResult.teacherIds;   // {frameId: true} for exclusion

    // Remove teacher frames from the student portrait pool
    portraitFrames = separateTeacherFrames(portraitFrames, teacherIdMap);

    // Primary teacher frame = slot 0 (T_Photo_01)
    var teacherFrame = (teacherSlots.length > 0) ? teacherSlots[0] : null;
    if (teacherFrame) {
        $.writeln("✓ Teacher portrait frame: FOUND (T_Photo_01)");
    } else {
        $.writeln("⚠ Teacher portrait frame: NOT FOUND — T_Photo_01 missing from doc.rectangles");
    }
    if (teacherSlots.length > 1) {
        $.writeln("  Additional teacher slots (will stay blank): " + (teacherSlots.length - 1));
    }

    // ── Count actual teachers from manifest ──────────────────────────────────
    var numActualTeachers = 0;
    if (manifest) {
        var entriesForCount = manifest.entries || [];
        for (var ec = 0; ec < entriesForCount.length; ec++) {
            if (entriesForCount[ec].is_teacher || entriesForCount[ec].number === 0) {
                numActualTeachers++;
            }
        }
    }
    var slotsToExtract = Math.min(numActualTeachers, teacherSlots.length);
    $.writeln("  Actual teachers: " + numActualTeachers +
              " → populating " + slotsToExtract + " slot(s)");

    // ── Find teacher ID plate frames directly by name (T_Card_01, T_Card_02…) ──
    var tCardMap = {};
    var studentIdPlateFrames = [];
    for (var ic = 0; ic < idPlateFrames.length; ic++) {
        var icName = getFrameName(idPlateFrames[ic]).toLowerCase();
        var icMatch = icName.match(/^t[_ ]?card[_ ]?(\d+)$/) ||
                      icName.match(/^t[_ ]card[_ ]?(\d+)$/);
        if (icMatch) {
            var icSlot = parseInt(icMatch[1], 10) - 1;
            tCardMap[icSlot] = idPlateFrames[ic];
            $.writeln("  T_Card slot " + icSlot + ": " + getFrameName(idPlateFrames[ic]));
        } else {
            studentIdPlateFrames.push(idPlateFrames[ic]);
        }
    }
    // Also try direct name lookup for T_Card frames (same robustness as T_Photo)
    if (!objectHasKeys(tCardMap)) {
        for (var cn = 1; cn <= 9; cn++) {
            var csuffix = (cn < 10 ? "0" : "") + cn;
            var ccands = ["T_Card_" + csuffix, "T_Card_" + cn, "T_Card" + csuffix];
            for (var cci = 0; cci < ccands.length; cci++) {
                try {
                    var cr = doc.rectangles.itemByName(ccands[cci]);
                    if (cr && cr.isValid) {
                        tCardMap[cn - 1] = cr;
                        $.writeln("  T_Card slot " + (cn-1) + " found by direct lookup: " + ccands[cci]);
                        break;
                    }
                } catch(e) {}
            }
        }
        // Rebuild studentIdPlateFrames excluding any T_Card frames found above
        if (objectHasKeys(tCardMap)) {
            var tCardIds = {};
            for (var tck in tCardMap) {
                if (tCardMap.hasOwnProperty(tck)) {
                    try { tCardIds[tCardMap[tck].id] = true; } catch(e) {}
                }
            }
            studentIdPlateFrames = [];
            for (var icr = 0; icr < idPlateFrames.length; icr++) {
                try { if (tCardIds[idPlateFrames[icr].id]) continue; } catch(e) {}
                studentIdPlateFrames.push(idPlateFrames[icr]);
            }
        }
    }
    idPlateFrames = studentIdPlateFrames;

    var teacherIdFrames = [];   // parallel array to teacherSlots
    for (var ts = 0; ts < teacherSlots.length; ts++) {
        if (ts < slotsToExtract && tCardMap.hasOwnProperty(ts)) {
            teacherIdFrames.push(tCardMap[ts]);
            $.writeln("✓ Teacher ID plate frame assigned for slot " + ts + ": T_Card_" + (ts + 1 < 10 ? "0" : "") + (ts + 1));
        } else {
            teacherIdFrames.push(null);
            if (ts < slotsToExtract) {
                $.writeln("  ⚠ No T_Card frame found for teacher slot " + ts);
            } else {
                $.writeln("  Teacher slot " + ts + " is empty — T_Card frame left untouched");
            }
        }
    }
    var teacherIdFrame = (teacherIdFrames.length > 0) ? teacherIdFrames[0] : null;

//     // ── Separate teacher name frames by name (T_Label_01, T_Label_02, …) ────
//     var tLabelMap = {};
//     var studentNameFrames = [];
//     for (var nl = 0; nl < nameFrames.length; nl++) {
//         var nlName = getFrameName(nameFrames[nl]).toLowerCase();
//         var nlMatch = nlName.match(/^t[_ ]?label[_ ]?(\d+)$/) ||
//                       nlName.match(/^t[_ ]label[_ ]?(\d+)$/);
//         if (nlMatch) {
//             var nlSlot = parseInt(nlMatch[1], 10) - 1;
//             tLabelMap[nlSlot] = nameFrames[nl];
//             $.writeln("  T_Label slot " + nlSlot + ": " + getFrameName(nameFrames[nl]));
//         } else {
//             studentNameFrames.push(nameFrames[nl]);
//         }
//     }
//     // Also try direct name lookup for T_Label text frames
//     if (!objectHasKeys(tLabelMap)) {
//         for (var tn = 1; tn <= 9; tn++) {
//             var tsuffix = (tn < 10 ? "0" : "") + tn;
//             var tcands = ["T_Label_" + tsuffix, "T_Label_" + tn, "T_Label" + tsuffix];
//             for (var tci3 = 0; tci3 < tcands.length; tci3++) {
//                 try {
//                     var tr = doc.textFrames.itemByName(tcands[tci3]);
//                     if (tr && tr.isValid) {
//                         tLabelMap[tn - 1] = tr;
//                         $.writeln("  T_Label slot " + (tn-1) + " found by direct lookup: " + tcands[tci3]);
//                         break;
//                     }
//                 } catch(e) {}
//             }
//         }
//         // Rebuild studentNameFrames excluding any T_Label frames found
//         if (objectHasKeys(tLabelMap)) {
//             var tLabelIds = {};
//             for (var tlk in tLabelMap) {
//                 if (tLabelMap.hasOwnProperty(tlk)) {
//                     try { tLabelIds[tLabelMap[tlk].id] = true; } catch(e) {}
//                 }
//             }
//             studentNameFrames = [];
//             for (var nlr = 0; nlr < nameFrames.length; nlr++) {
//                 try { if (tLabelIds[nameFrames[nlr].id]) continue; } catch(e) {}
//                 studentNameFrames.push(nameFrames[nlr]);
//             }
//         }
//     }

//     var teacherNameFrames = [];
//     for (var ts2 = 0; ts2 < teacherSlots.length; ts2++) {
//         if (ts2 < slotsToExtract && tLabelMap.hasOwnProperty(ts2)) {
//             teacherNameFrames.push(tLabelMap[ts2]);
//             $.writeln("✓ Teacher name frame assigned for slot " + ts2 + ": T_Label_" + (ts2 + 1 < 10 ? "0" : "") + (ts2 + 1));
//         } else {
//             teacherNameFrames.push(null);
//             if (ts2 < slotsToExtract) {
//                 $.writeln("  ⚠ No T_Label frame found for teacher slot " + ts2);
//             } else {
//                 $.writeln("  Teacher slot " + ts2 + " is empty — T_Label frame left untouched");
//             }
//         }
//     }
//     var teacherNameFrame = (teacherNameFrames.length > 0) ? teacherNameFrames[0] : null;

    // ── Sort remaining (student) frames ──
    portraitFrames = sortFramesByPosition(portraitFrames);
    idPlateFrames  = sortFramesByPosition(idPlateFrames);
    // nameFrames  = sortFramesByPosition(nameFrames); // [NAMES]

    $.writeln("✓ Student portrait frames: " + portraitFrames.length);

    // [NAMES] frameNames holds the name string for each student frame.
    // var frameNames = [];
    // for (var fi = 0; fi < nameFrames.length; fi++) { frameNames[fi] = ""; }
    var frameNames = []; // [NAMES] empty placeholder

    // ── Place group photo ──
    if (groupFrame && CONFIG.groupPhotoFile) {
        var gpFile = new File(CONFIG.groupPhotoFile);
        if (gpFile.exists) {
            groupFrame.place(gpFile);
            groupFrame.fit(FitOptions.FILL_PROPORTIONALLY);
            $.writeln("✓ Group photo placed");
        }
    }

    // ── Load absent list from file (teammate update) ──
    var absentList = readAbsentNumbers(CONFIG.absentFile);

    // ── Place teacher from manifest entry (number 0 / is_teacher) ──
    var teacherEntry = null;
    if (manifest) {
        var entries2 = manifest.entries || [];
        for (var te = 0; te < entries2.length; te++) {
            if (entries2[te].is_teacher || entries2[te].number === 0) {
                teacherEntry = entries2[te];
                break;
            }
        }
    }

    var teacherPlacement;

    if (CONFIG.ignoreTeacherManifest) {
        // Ignore manifest offsets → use CONFIG
        teacherPlacement = {
            offsetX: CONFIG.teacher.offsetX,
            offsetY: CONFIG.teacher.offsetY,
            scaleFactor: CONFIG.teacher.scaleFactor,
            source: "CONFIG.teacher (ignore manifest)"
        };
    } else {
        // Use manifest if available
        teacherPlacement = resolveFacePlacement(teacherEntry, CONFIG.teacher);
    }

    // ── Diagnostic: dump what teacherEntry looks like ──
    if (teacherEntry) {
        $.writeln("  Teacher entry found: number=" + teacherEntry.number +
                  " name=" + teacherEntry.name + " is_teacher=" + teacherEntry.is_teacher);
        var _tFiles = ""; for (var _k in teacherEntry.files) {
            if (teacherEntry.files.hasOwnProperty(_k)) _tFiles += _k + ":" + teacherEntry.files[_k] + " ";
        }
        $.writeln("  Teacher files: " + _tFiles);
        $.writeln("  teacherFrame is " + (teacherFrame ? "SET" : "NULL"));
    } else {
        $.writeln("⚠ No teacher entry found in manifest (is_teacher or number===0)");
    }

    if (teacherFrame && teacherEntry && teacherEntry.files) {
        // Portrait — accept both old (本_01) and new (本01) key formats,
        // picking the lowest-numbered available file.
        var tPortrait = teacherEntry.files["本_01"] ||
                        teacherEntry.files["本01"]  ||
                        teacherEntry.files["本_02"] ||
                        teacherEntry.files["本02"]  ||
                        teacherEntry.files["本_03"] ||
                        teacherEntry.files["本03"];
        $.writeln("  Teacher portrait key resolved: " + (tPortrait || "NONE"));
        if (tPortrait) {
            var tPortraitFile = resolveTeacherFile(tPortrait, CONFIG.packageFolder);
            if (tPortraitFile.exists) {
                placeWithTransform(teacherFrame, tPortraitFile,
                    teacherPlacement.offsetX, teacherPlacement.offsetY, teacherPlacement.scaleFactor);
                $.writeln("✓ Teacher portrait placed: " + tPortrait);
                $.writeln("  -> Teacher offsets from " + teacherPlacement.source +
                          " (x=" + teacherPlacement.offsetX +
                          ", y=" + teacherPlacement.offsetY +
                          ", scale=" + teacherPlacement.scaleFactor + "%)");
            } else {
                $.writeln("⚠ Teacher portrait not found — tried paths logged above. File: " + tPortrait);
                $.writeln("  packageFolder (normalized): " + normalizePath(CONFIG.packageFolder));
            }
        } else {
            $.writeln("⚠ Teacher files object has no 本_01 / 本01 key — available keys:");
            for (var tk in teacherEntry.files) {
                if (teacherEntry.files.hasOwnProperty(tk)) $.writeln("    '" + tk + "': " + teacherEntry.files[tk]);
            }
        }
        // ID plate — accept both 札 and 札01 key formats
        var tIdFileName = teacherEntry.files["札"] || teacherEntry.files["札01"];
        if (teacherIdFrame && tIdFileName) {
            var tIdFile = resolveTeacherFile(tIdFileName, CONFIG.packageFolder);
            if (tIdFile.exists) {
                teacherIdFrame.place(tIdFile);
                smartFit(teacherIdFrame, CONFIG.idPlate.scaleFactor,
                         CONFIG.idPlate.offsetX, CONFIG.idPlate.offsetY);
                $.writeln("✓ Teacher ID plate placed (offset Y=" + CONFIG.idPlate.offsetY + "mm)");
            } else {
                $.writeln("⚠ Teacher ID plate file not found — tried paths logged above. File: " + tIdFileName);
            }
        }
//         // Name
//         if (teacherNameFrame) {
//             var tName = teacherEntry.name || "";
//             try {
//                 // Sample paragraph style point size BEFORE overwriting
//                 // (template may have character-level size overrides that cause mixed sizes)
//                 var tTargetSize = null;
//                 try {
//                     var tPS = teacherNameFrame.paragraphs[0].appliedParagraphStyle;
//                     var tPSSize = tPS.pointSize;
//                     if (tPSSize && tPSSize > 0) tTargetSize = tPSSize;
//                 } catch (e) {}
//                 // Fallback: use the minimum size found (treat larger chars as accidental overrides)
//                 if (!tTargetSize) {
//                     try {
//                         var tMinSize = Infinity;
//                         var tExistChars = teacherNameFrame.characters;
//                         for (var tci = 0; tci < tExistChars.length; tci++) {
//                             var tChSz = tExistChars[tci].pointSize;
//                             if (tChSz > 0 && tChSz < tMinSize) tMinSize = tChSz;
//                         }
//                         if (tMinSize < Infinity) tTargetSize = tMinSize;
//                     } catch (e) {}
//                 }

//                 // Set the name content
//                 teacherNameFrame.contents = tName;

//                 // Normalize: stamp every character with the same point size
//                 if (tTargetSize) {
//                     var tNameChars = teacherNameFrame.characters;
//                     for (var tci2 = 0; tci2 < tNameChars.length; tci2++) {
//                         try { tNameChars[tci2].pointSize = tTargetSize; } catch (e) {}
//                     }
//                     $.writeln("  ✓ Teacher name font size normalized to " + tTargetSize + "pt");
//                 }

//                 centerNameFrameText(teacherNameFrame);

//                 $.writeln("✓ Teacher name set: " + tName);
//             } catch (e) {
//                 $.writeln("  ✗ Teacher name error: " + e.message);
//             }
//         }
    } else if (teacherFrame && CONFIG.teacherPhotoFile) {
        // Fallback: use the manually configured file path
        var tFile2 = new File(CONFIG.teacherPhotoFile);
        if (tFile2.exists) {
            placeWithTransform(teacherFrame, tFile2,
                CONFIG.teacher.offsetX, CONFIG.teacher.offsetY, CONFIG.teacher.scaleFactor);
            $.writeln("✓ Teacher photo placed (CONFIG path)");
        }
    } else {
        $.writeln("⚠ No teacher entry found in manifest and CONFIG.teacherPhotoFile is empty — teacher frame left blank");
    }

    // ── Place student photos ──
    var placed = 0;
    var absent = 0;
    var frameIdx = 0;

    // Decide iteration strategy:
    //   (a) Roster known  → iterate 1..total_students (fills gaps with "absent")
    //   (b) Roster missing/zero (e.g. partial batch with sparse numbering like
    //       A2, A11, A17) → iterate only the students we actually have, packing
    //       them into the leading frames in numeric order.
    var rosterKnown = manifest && manifest.total_students > 0;
    var iterList;
    if (rosterKnown) {
        iterList = [];
        for (var s = 1; s <= manifest.total_students; s++) iterList.push(s);
    } else {
        iterList = studentNums.slice(); // already sorted ascending
        $.writeln("  ℹ Roster size unknown (total_students=0) — packing " +
                  iterList.length + " student(s) into leading frames");
    }

    for (var ii = 0; ii < iterList.length; ii++) {
        var studentNum = iterList[ii];
        var key = String(studentNum);
        var data = fileMap[key];
        var studentName = nameMap[key] || "";
        var studentEntry = entryMap[key] || null;
        var studentPlacement = resolveFacePlacement(studentEntry, CONFIG.student);

        if (frameIdx >= portraitFrames.length) {
            $.writeln("⚠ Ran out of portrait frames at student #" + studentNum);
            break;
        }

        // Check absent via absent.txt (teammate update) OR via missing files
        // Support both old format (本_01, 札) and new format (本01, 札01)
        var hasPortrait = data && (data["本_01"] || data["本01"]);
        var hasIdPlate = data && (data["札"] || data["札01"]);
        if (isAbsent(studentNum, absentList) || !data || (!hasPortrait && !hasIdPlate)) {
            $.writeln("⊗ Student #" + studentNum + " (" + studentName + "): ABSENT — photo frames filled with absent image, name kept");
            highlightAbsentFrame(portraitFrames[frameIdx]);  // Use updated function to place absent image in portrait frame

            if (frameIdx < idPlateFrames.length) {
                highlightAbsentFrame(idPlateFrames[frameIdx]);  // Use updated function to place absent image in ID plate frame
            }

            // [NAMES] assign name even for absent students
            // if (frameIdx < nameFrames.length) {
            //     frameNames[frameIdx] = studentName;
            // }

            absent++;
            frameIdx++;   // advance frame (portrait/ID left as absent image)
            continue;
        }

        // ── Portrait (本_01 or 本01) → Default layer — now uses placeWithTransform ──
        var portraitFile = data["本_01"] || data["本01"];
        if (portraitFile) {
            placeWithTransform(
                portraitFrames[frameIdx], portraitFile,
                studentPlacement.offsetX,
                studentPlacement.offsetY,
                studentPlacement.scaleFactor
            );
            $.writeln("✓ #" + studentNum + " portrait placed");
            $.writeln("  -> #" + studentNum + " offsets from " + studentPlacement.source +
                      " (x=" + studentPlacement.offsetX +
                      ", y=" + studentPlacement.offsetY +
                      ", scale=" + studentPlacement.scaleFactor + "%)");
        }

        // ── ID Plate (札 or 札01) → 札持ちカット layer — now with offset support ──
        var idPlateFile = data["札"] || data["札01"];
        if (idPlateFile && frameIdx < idPlateFrames.length) {
            idPlateFrames[frameIdx].place(idPlateFile);
            smartFit(idPlateFrames[frameIdx], CONFIG.idPlate.scaleFactor,
                     CONFIG.idPlate.offsetX, CONFIG.idPlate.offsetY);
            $.writeln("  ✓ #" + studentNum + " ID plate placed (offset Y=" + CONFIG.idPlate.offsetY + "mm)");
        }

        // [NAMES] collect name for batch write — re-enable with placeNamesInFrames
        // if (frameIdx < nameFrames.length) {
        //     frameNames[frameIdx] = studentName;
        // }

        placed++;
        frameIdx++;
    }

    // [NAMES] ── Write all names to text frames ──
    // $.writeln("Writing names to " + nameFrames.length + " name frames…");
    // placeNamesInFrames(nameFrames, frameNames);

    // ── Save ──
    if (CONFIG.autoSave && CONFIG.outputFile) {
        var saveFile = new File(CONFIG.outputFile);
        doc.save(saveFile);
        $.writeln("✓ Saved: " + CONFIG.outputFile);
    }

    if (CONFIG.autoClose) {
        doc.close(SaveOptions.NO);
    }

    // Summary
    var summary =
        "\n════════════════════════════════════════\n" +
        "✓ AutoPlacePhotos v14 COMPLETE\n" +
        "════════════════════════════════════════\n" +
        "  Class    : " + CONFIG.classLetter + "\n" +
        "  Placed   : " + placed + "\n" +
        "  Absent   : " + absent + "\n" +
        "  Teacher  : " + (CONFIG.teacherPhotoFile ? "yes" : "—") + "\n" +
        "  Group    : " + (CONFIG.groupPhotoFile ? "yes" : "—") + "\n" +
        "  Output   : " + (CONFIG.outputFile || "not saved") + "\n";

    $.writeln(summary);
    alert(summary);

})();