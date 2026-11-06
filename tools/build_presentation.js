// Team 9 proposal presentation as .pptx (same content and look as the online deck).
// npm install pptxgenjs; NODE_PATH=<its node_modules> PPTX_SKILL=<pptx skill dir> node tools/build_presentation.js . docs/presentation/njit_logo.png docs/presentation/Team9_Proposal_Presentation.pptx
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const [ROOT, LOGO, OUT] = process.argv.slice(2);
const FIG = (f) => path.join(ROOT, "docs", "figures", f);

const INK = "1C1B19", PAPER = "EFEBE2", CARD = "FBF9F4", YEL = "F4D35E", BLUE = "2944C2";
const MUTED = "4A4741", MUTED_D = "B9B2A4", BAND = "E6E0D3", RULE = "B7AF9F", NINE = "E2DCCF";
const SERIF = "Cambria", SANS = "Calibri", MONO = "Courier New";
const THEME = {
  name: "Sign-to-Speech", headFontFace: SERIF, bodyFontFace: SANS,
  colors: { dk1: INK, lt1: CARD, dk2: MUTED, lt2: PAPER, accent1: YEL, accent2: BLUE, accent3: MUTED_D,
            accent4: BAND, accent5: "3D3A35", accent6: RULE, hlink: BLUE, folHlink: MUTED },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: SERIF, bodyFontFace: SANS };
pres.title = "Sign-to-Speech Communication Device: Proposal";
pres.author = "Team 9";

const X0 = 0.85, CW = 11.63; // margins and content width
const footer = (dark, text) => [
  { text: { text, options: { x: X0, y: 6.9, w: 9, h: 0.3, margin: 0, fontFace: MONO, fontSize: 11, charSpacing: 1,
                                 color: dark ? MUTED_D : MUTED } } },
  { rect: { x: 11.88, y: 6.86, w: 0.6, h: 0.36, fill: { color: YEL }, line: { color: YEL } } },
];
const slideNum = { x: 11.88, y: 6.86, w: 0.6, h: 0.36, fontFace: MONO, fontSize: 13, bold: true, color: INK, align: "center",
                   valign: "middle" };
const label = (dark) => ({ placeholder: { options: { name: "label", type: "body", x: X0, y: 0.62, w: CW, h: 0.34, margin: 0,
  fontFace: MONO, fontSize: 12, charSpacing: 2, color: dark ? YEL : BLUE, valign: "top" }, text: "" } });
const title = (dark, h) => ({ placeholder: { options: { name: "title", type: "title", x: X0, y: 1.0, w: CW, h, margin: 0,
  fontFace: SERIF, fontSize: 40, color: dark ? PAPER : INK, valign: "top", align: "left" }, text: "" } });

pres.defineSlideMaster({ title: "LIGHT", background: { color: PAPER },
  objects: [...footer(false, "TEAM 9 · SIGN-TO-SPEECH COMMUNICATION DEVICE"), label(false), title(false, 0.85)], slideNumber: slideNum });
pres.defineSlideMaster({ title: "DARK", background: { color: INK },
  objects: [...footer(true, "TEAM 9 · SIGN-TO-SPEECH COMMUNICATION DEVICE"), label(true), title(true, 1.45)], slideNumber: slideNum });
pres.defineSlideMaster({ title: "COVER", background: { color: PAPER },
  objects: [...footer(false, "HELEN AND JOHN C. HARTMANN DEPT. OF ECE")], slideNumber: slideNum });

const T = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, margin: 0, valign: "top" }, o));
const hline = (s, x, y, w, color, pt) => s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width: pt } });
const vline = (s, x, y, h, color, pt) => s.addShape(pres.shapes.LINE, { x, y, w: 0, h, line: { color, width: pt } });
const notes = (s, t) => s.addNotes(t);

// table helper: rows of strings; first row is the header
function table(s, rows, o) {
  const n = rows.length;
  const data = rows.map((r, i) => r.map((cell, j) => {
    const none = { type: "none" };
    const top = i === 0 ? { type: "solid", pt: 1.5, color: o.ink } : (i === 1 ? { type: "solid", pt: 0.75, color: o.ink } : none);
    const bottom = i === n - 1 ? { type: "solid", pt: 1.5, color: o.ink } : none;
    const fill = o.rowFill && o.rowFill(i) ? { color: o.rowFill(i) } : undefined;
    const color = (o.cellColor && o.cellColor(i, j)) || o.text;
    return { text: cell, options: { bold: i === 0 || (o.boldRow && o.boldRow(i)), color, fill, border: [top, none, bottom, none] } };
  }));
  s.addTable(data, { x: o.x, y: o.y, w: o.w, colW: o.colW, rowH: o.rowH, fontFace: SANS, fontSize: o.fontSize,
                     margin: [3, 6, 3, 6], valign: "middle", autoPage: false });
}

// ---------------------------------------------------------------- 1 cover
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "COVER", sectionTitle: "Introduction" });
T(s, "9", { x: 8.3, y: -0.35, w: 4.4, h: 6.0, fontFace: SERIF, fontSize: 380, color: NINE, align: "right", objectName: "team number" });
s.addImage({ path: LOGO, x: X0, y: 0.55, w: 1.9, h: 0.5, altText: "NJIT logo" });
T(s, "ECE 414 · PROPOSAL · FALL 2026", { x: 7.5, y: 0.68, w: 4.98, h: 0.3, fontFace: MONO, fontSize: 12, charSpacing: 2,
  color: MUTED, align: "right" });
T(s, "Sign‑to‑Speech", { x: X0, y: 1.45, w: 9.5, h: 1.35, fontFace: SERIF, fontSize: 80, color: INK });
T(s, "A multilingual communication device: edge-deployed ASL recognition with confidence-gated speech output",
  { x: X0, y: 2.9, w: 8.0, h: 0.85, fontFace: SANS, fontSize: 18, color: MUTED, lineSpacingMultiple: 1.15 });
s.addShape(pres.shapes.RECTANGLE, { x: X0, y: 3.95, w: 6.55, h: 0.6, fill: { color: YEL }, line: { color: YEL } });
T(s, "Signs in. Speech out. Nothing leaves the device.", { x: X0 + 0.15, y: 3.95, w: 6.3, h: 0.6, valign: "middle",
  fontFace: SERIF, fontSize: 22, italic: true, color: INK });
T(s, "TEAM 9", { x: X0, y: 4.95, w: 3, h: 0.3, fontFace: MONO, fontSize: 12, bold: true, charSpacing: 2, color: INK });
const team = [["01", "Asmar Hasanova", "Sign recognition"], ["02", "Dorukhan Cakir", "Sentence and confirmation"],
              ["03", "Zeynep Hafsa Cakici", "Translation, speech, reply"], ["04", "Merrick Simmons", "Embedded system"]];
team.forEach(([n, name, role], i) => {
  const x = X0 + i * (2.75 + 0.21);
  hline(s, x, 5.35, 2.75, INK, 1.5);
  T(s, n, { x, y: 5.45, w: 2.75, h: 0.25, fontFace: MONO, fontSize: 12, color: BLUE });
  T(s, name, { x, y: 5.72, w: 2.75, h: 0.32, fontFace: SANS, fontSize: 15, bold: true, color: INK });
  T(s, role, { x, y: 6.05, w: 2.75, h: 0.3, fontFace: SANS, fontSize: 12.5, color: MUTED });
});
notes(s, "Good morning, we are Team 9. Our project is a sign-to-speech communication device: a Deaf person signs to a camera, and the device speaks the sentence aloud in English, Turkish or Spanish. The team is listed in the order the data flows through the device: Asmar recognizes the signs, Dorukhan turns them into a sentence the user confirms, Zeynep translates and speaks it and brings the reply back, and Merrick makes it all run on one embedded board. (About 20 seconds.)");

// ---------------------------------------------------------------- 2 abstract
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
s.addText("02 / ABSTRACT", { placeholder: "label" });
s.addText([{ text: "It speaks only when the signer says " }, { text: "yes", options: { italic: true, color: YEL } }],
  { placeholder: "title" });
T(s, "A Deaf user signs to one ordinary camera. Our recognizer reads the signs and how sure it is, the user confirms the sentence on a touchscreen, and only then does the device speak it in English, Turkish or Spanish. The hearing person’s reply comes back as text. Everything runs offline on an NVIDIA Jetson.",
  { x: X0, y: 2.05, w: 9.8, h: 1.75, fontFace: SANS, fontSize: 17, color: "C8C1B3", lineSpacingMultiple: 1.2 });
const steps = [["1", "Sign", "one camera"], ["2", "Landmarks", "MediaPipe"], ["3", "Recognize", "with confidence"],
               ["4", "Confirm", "touchscreen"], ["5", "Translate", "NLLB-200"], ["6", "Speak", "Piper voices"]];
const sw = CW / 6, sy = 4.3;
hline(s, X0, sy, CW, PAPER, 1.5);
steps.forEach(([n, w, sub], i) => {
  const x = X0 + i * sw, hot = i === 3;
  if (hot) s.addShape(pres.shapes.RECTANGLE, { x, y: sy + 0.02, w: sw, h: 1.45, fill: { color: YEL }, line: { color: YEL } });
  else if (i > 0 && i !== 4) vline(s, x, sy + 0.12, 1.2, MUTED, 0.75);
  if (i === 4) vline(s, x + sw, sy + 0.12, 0, MUTED, 0);
  const pad = i === 0 ? 0 : 0.15;
  T(s, n, { x: x + pad, y: sy + 0.15, w: sw - 0.3, h: 0.25, fontFace: MONO, fontSize: 12, color: hot ? INK : MUTED_D });
  T(s, w, { x: x + pad, y: sy + 0.45, w: sw - 0.2, h: 0.5, fontFace: SERIF, fontSize: 23, color: hot ? INK : PAPER });
  T(s, sub, { x: x + pad, y: sy + 0.98, w: sw - 0.25, h: 0.3, fontFace: SANS, fontSize: 12.5, color: hot ? INK : MUTED_D });
});
notes(s, "In one sentence: the device speaks for a Deaf user, but only after they say yes to the sentence. Walk the six steps left to right: one ordinary camera, MediaPipe body and hand landmarks, our recognizer, which also reports how sure it is, then the confirmation on the touchscreen, highlighted because it is the heart of the design, then translation with NLLB-200 and speech with Piper. The hearing person’s reply comes back through Whisper as English text. Everything runs offline, so no video ever leaves the device. It is an aid for when no interpreter is available, not a replacement. (About 60 seconds.)");

// ---------------------------------------------------------------- 3 specs
pres.addSection({ title: "Requirements" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Requirements" });
s.addText("03 / DESIGN OBJECTIVES & SPECIFICATIONS", { placeholder: "label" });
s.addText("Eight numbers we hold ourselves to", { placeholder: "title" });
table(s, [["Objective", "Target", "Today", "Measured on"],
  ["Top-1 accuracy, unseen signers", "≥ 75%", "86.6%", "ASL Citizen, 11 Deaf signers"],
  ["Top-5 accuracy, unseen signers", "≥ 90%", "98.2%", "ASL Citizen, 11 Deaf signers"],
  ["Confirmation to speech", "< 1.2 s", "0.71–0.73 s", "Cloud CPU; Jetson next"],
  ["Frame rate", "≥ 15 fps", "≈ 20 fps", "Laptop; Jetson next"],
  ["Memory, all models", "≤ 8 GB", "≈ 4.4 GB", "Estimate from model files"],
  ["Vocabulary", "100–250 signs", "134 words", "Draft; review with Deaf users"],
  ["Spoken languages", "EN, TR, ES", "all 3 run", "Offline, on a CPU"],
  ["Network", "none needed", "offline", "Wi-Fi card removed by design"]],
  { x: X0, y: 2.0, w: CW, colW: [3.75, 2.05, 2.2, 3.63], rowH: 0.47, fontSize: 15, ink: INK, text: INK,
    rowFill: (i) => (i > 0 && i % 2 === 0 ? BAND : null), cellColor: (i, j) => (i > 0 && j === 2 ? BLUE : null) });
notes(s, "Our design objectives as eight measurable specifications, with where we stand today and what each number was measured on. Recognition: at least 75 percent top-1 and 90 percent top-5 on signers the model has never seen; our v5 model reaches 86.6 and 98.2 percent on the 11 Deaf test signers of the ASL Citizen dataset. Speech must start within 1.2 seconds of confirmation; translation plus the first audio already takes about 0.7 seconds on an ordinary CPU. All models together are estimated at 4.4 of the 8 gigabytes. The right-hand column is the honest part: several numbers are from a laptop or a cloud CPU, and the Jetson and our own camera come next. If asked: the 86.6 percent is on a 134-word vocabulary from recorded videos, not yet live. (About 75 seconds.)");

// ---------------------------------------------------------------- 4 constraints
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Requirements" });
s.addText("04 / CONSTRAINTS & STANDARDS", { placeholder: "label" });
s.addText("A small box, a student budget, strict rules", { placeholder: "title" });
const cons = [["Technical", "8 GB shared memory, 1.2 s budget, one RGB camera, no network"],
  ["Economic", "≈ $495–665 in parts, split four ways; free, open-source models"],
  ["Legal", "ADA: an aid, never a replacement for interpreters; non-commercial data and model licences; IRB check for recordings"],
  ["Ethical", "Nothing spoken without the user’s confirmation; accuracy reported for every signer; Deaf users involved"],
  ["Time, safety", "Propose in ECE 414, build in ECE 416; heat, power, safe touch temperature"]];
const cy0 = 2.05, ch = 0.86, cw = 6.85;
cons.forEach(([k, v], i) => {
  const y = cy0 + i * ch;
  hline(s, X0, y, cw, i === 0 ? INK : RULE, i === 0 ? 1.5 : 0.75);
  T(s, k, { x: X0, y: y + 0.12, w: 1.75, h: 0.45, fontFace: SERIF, fontSize: 20, color: INK });
  T(s, v, { x: X0 + 1.85, y: y + 0.12, w: cw - 1.85, h: ch - 0.18, fontFace: SANS, fontSize: 13.5, color: "3D3A35" });
});
hline(s, X0, cy0 + 5 * ch, cw, INK, 1.5);
const px = 8.15, pw = X0 + CW - px;
s.addShape(pres.shapes.RECTANGLE, { x: px, y: cy0, w: pw, h: 5 * ch, fill: { color: INK }, line: { color: INK } });
T(s, "STANDARDS WE FOLLOW", { x: px + 0.3, y: cy0 + 0.25, w: pw - 0.6, h: 0.3, fontFace: MONO, fontSize: 12, charSpacing: 2, color: YEL });
const stds = [["IEEE 7000-2021", "ethics in system design"], ["IEEE 7002-2022", "data privacy process"],
  ["IEEE 1012-2024", "verification and validation"], ["IEEE 2841-2022", "deep-learning evaluation"],
  ["ANSI/UL 62368-1", "equipment safety"], ["NIST AI 100-1", "AI risk management"]];
stds.forEach(([c, d], i) => {
  const y = cy0 + 0.68 + i * 0.6;
  hline(s, px + 0.3, y, pw - 0.6, MUTED, 0.75);
  T(s, c, { x: px + 0.3, y: y + 0.06, w: pw - 0.6, h: 0.26, fontFace: MONO, fontSize: 13, bold: true, color: PAPER });
  T(s, d, { x: px + 0.3, y: y + 0.31, w: pw - 0.6, h: 0.25, fontFace: SANS, fontSize: 12.5, color: MUTED_D });
});
notes(s, "Our constraints come in five groups. Technical: everything must fit in the Jetson’s 8 gigabytes and meet the 1.2 second budget with one ordinary camera and no network. Economic: about 500 to 665 dollars, paid by the four of us, so only free, open-source models. Legal: under the ADA the device must never be presented as a replacement for an interpreter; the dataset and translation model are licensed for non-commercial use only; recordings of people need consent and an IRB check. Ethical: nothing is spoken without the user’s confirmation, and we report accuracy for each signer. On the right, the six standards: IEEE 7000 for ethical design, 7002 for privacy, 1012 for verification and validation, 2841 for evaluating deep-learning models, UL 62368-1 for electrical and thermal safety, and the NIST AI risk management framework. (About 75 seconds.)");

// ---------------------------------------------------------------- 5 block diagram
pres.addSection({ title: "Design" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Design" });
s.addText("05 / COMPONENTS & BLOCK DIAGRAM", { placeholder: "label" });
s.addText("One $250 board does all the thinking", { placeholder: "title" });
const bx = X0, by = 2.0, bw = 4.75, bh = 4.6;
s.addShape(pres.shapes.RECTANGLE, { x: bx, y: by, w: bw, h: bh, fill: { color: CARD }, line: { color: INK, width: 0.75 } });
const iw = bw - 0.2, ih = iw * 1436 / 1513;
s.addImage({ path: FIG("fig_hw_block_diagram.png"), x: bx + 0.1, y: by + (bh - ih) / 2, w: iw, h: ih,
  altText: "Hardware block diagram: camera, touchscreen, button and LED face the Deaf user; speaker and microphone face the hearing person; Jetson Orin Nano developer kit with SSD and power inside the base; Wi-Fi removed" });
const tx = bx + bw + 0.45, tw = X0 + CW - tx;
table(s, [["Component", "Role", "USD"], ["Jetson Orin Nano 8 GB kit", "runs every model", "250"],
  ["USB camera, 1080p", "sees the signer", "40–70"], ["7-inch touchscreen", "confirm, read reply", "60–90"],
  ["USB microphone, speaker", "hearing side", "35–70"], ["256 GB NVMe SSD", "OS and models", "30–40"],
  ["Enclosure, cables, controls", "3D-printed base", "30–65"], ["Battery pack (optional)", "portable use", "50–80"],
  ["Total", "", "495–665"]],
  { x: tx, y: by, w: tw, colW: [3.05, 2.25, tw - 5.3], rowH: 0.42, fontSize: 14, ink: INK, text: INK,
    rowFill: (i) => (i === 8 ? YEL : (i > 0 && i % 2 === 0 ? BAND : null)), boldRow: (i) => i === 8 });
T(s, "On the module: MediaPipe · our recognizer (1.78 M parameters, about 4 ms per sign) · NLLB-200 · Piper · Whisper",
  { x: tx, y: by + 9 * 0.42 + 0.25, w: tw, h: 0.6, fontFace: SANS, fontSize: 13.5, color: "3D3A35" });
notes(s, "This is the hardware block diagram. Everything that thinks is one NVIDIA Jetson Orin Nano developer kit in the base. On the Deaf user’s side: the camera, the touchscreen, and a button and LED; the LED lights whenever the camera or microphone is on. On the hearing person’s side: speaker and microphone, all on USB. The board has only DisplayPort, so the touchscreen uses an adapter. We remove the Wi-Fi card and leave Ethernet unplugged, so offline is a physical fact you can check. Parts total about 495 to 665 dollars. On the module run MediaPipe, our 1.78-million-parameter recognizer, which needs about 4 milliseconds per sign, NLLB-200, Piper and Whisper. (About 75 seconds.)");

// ---------------------------------------------------------------- 6 prototyping
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Design" });
s.addText("06 / PROTOTYPING, VALIDATION, TESTING & ASSEMBLY", { placeholder: "label" });
s.addText("Cheap prototypes first, then the real device", { placeholder: "title" });
const fx = X0, fy = 2.0, fw = 4.6, fh = 4.2;
s.addShape(pres.shapes.RECTANGLE, { x: fx, y: fy, w: fw, h: fh, fill: { color: CARD }, line: { color: INK, width: 0.75 } });
const fiw = fw - 0.2, fih = fiw * 1355 / 1524;
s.addImage({ path: FIG("fig_design_flow.png"), x: fx + 0.1, y: fy + (fh - fih) / 2, w: fiw, h: fih,
  altText: "Course design flow chart filled in for our project: objectives, benchmarking, concept and constraints, business, three component designs, prototyping and testing, prototype assembly" });
T(s, "Course design flow, filled in", { x: fx, y: fy + fh + 0.1, w: fw, h: 0.28, fontFace: MONO, fontSize: 11, color: MUTED });
const rx = fx + fw + 0.45, rw = X0 + CW - rx, rh = 0.52;
const protos = [["P1", "Proof of principle:", " laptop and webcam live demo", "BUILT"],
  ["P2", "User experience:", " paper screens, Wizard of Oz", "NEXT"], ["P3", "Bench:", " Jetson with USB parts, no case", "PURCHASE"],
  ["P4", "Form study:", " 3D-printed enclosure", "PLANNED"], ["P5", "Working prototype:", " full device, assembled", "ECE 416"]];
protos.forEach(([c, b, rest, st], i) => {
  const y = fy + i * rh;
  hline(s, rx, y, rw, i === 0 ? INK : RULE, i === 0 ? 1.5 : 0.75);
  T(s, c, { x: rx, y: y + 0.12, w: 0.5, h: 0.3, fontFace: MONO, fontSize: 13, bold: true, color: INK });
  T(s, [{ text: b, options: { bold: true } }, { text: rest }], { x: rx + 0.55, y: y + 0.12, w: rw - 2.05, h: 0.32,
    fontFace: SANS, fontSize: 14, color: INK });
  if (i === 0) s.addShape(pres.shapes.RECTANGLE, { x: rx + rw - 1.4, y: y + 0.1, w: 1.4, h: 0.32, fill: { color: YEL }, line: { color: YEL } });
  T(s, st, { x: rx + rw - 1.4, y: y + 0.1, w: 1.4, h: 0.32, valign: "middle", align: "center", fontFace: MONO, fontSize: 12,
    color: i === 0 ? INK : BLUE });
});
hline(s, rx, fy + 5 * rh, rw, INK, 1.5);
T(s, [{ text: "Validated so far: ", options: { bold: true, color: BLUE } },
      { text: "86.6% top-1 on 11 unseen Deaf signers; the device answers 85.1% of signs at 6.5% error and asks the user about the rest." }],
  { x: rx, y: fy + 5 * rh + 0.25, w: rw, h: 0.8, fontFace: SANS, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.1 });
T(s, "UNIT → INTEGRATION → SYSTEM TESTS (R1–R15) → VALIDATION WITH DEAF SIGNERS",
  { x: rx, y: fy + 5 * rh + 1.2, w: rw, h: 0.6, fontFace: MONO, fontSize: 12, color: "3D3A35", lineSpacingMultiple: 1.15 });
notes(s, "On the left is the course’s design flow chart, filled in for our project: our three goals are components X, Y and Z, and we added one loop back to the concept when a test misses its target. We prototype from cheap to expensive. P1, the proof of principle, is built: a laptop and webcam running the full recognizer live. P2 tests the confirmation screen with Deaf users on paper, with a teammate playing the computer, the Wizard of Oz method. P3 is the Jetson on the bench, P4 the 3D-printed enclosure, and P5 the assembled device in ECE 416. Validation so far is on recorded data: 86.6 percent top-1, and at our threshold the device answers 85 percent of signs at 6.5 percent error and asks the user about the rest. Testing goes from unit to integration to 15 system requirements, ending with Deaf users at a mock counter. (About 80 seconds.)");

// ---------------------------------------------------------------- 7 benchmarking
pres.addSection({ title: "Comparison" });
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Comparison" });
s.addText("07 / BENCHMARKING · BUSINESS & ECONOMIC POTENTIAL", { placeholder: "label" });
s.addText([{ text: "Others need a cloud, gloves or depth cameras. " }, { text: "We need one webcam", options: { italic: true, color: YEL } }],
  { placeholder: "title" });
table(s, [["System", "Input", "Processing", "Abstains or confirms", "Output"],
  ["Sign-Speak, patent 2026", "one camera", "cloud server", "menu of 3 sentences", "text"],
  ["Avodah, patent 2019", "cameras + depth", "local or cloud", "not described", "speech, avatar"],
  ["Sorenson, patent 2024", "video call", "relay servers", "not described", "voice, text"],
  ["Bosch, patent 2018", "sensor gloves", "local", "not described", "text, audio"],
  ["Our device", "one RGB camera", "offline Jetson", "both, by confidence", "speech EN/TR/ES"]],
  { x: X0, y: 2.55, w: CW, colW: [2.75, 2.25, 2.0, 2.55, 2.08], rowH: 0.4, fontSize: 13.5, ink: PAPER, text: PAPER,
    rowFill: (i) => (i === 5 ? YEL : null), cellColor: (i) => (i === 5 ? INK : null), boldRow: (i) => i === 5 });
const stats = [["63.1% → 86.6%", "ASL Citizen baseline (2,731 signs) vs ours (134-word subset)"],
  ["$495–665", "parts, with no per-use or cloud fees"],
  ["69%", "of surveyed Deaf users accept a stand-alone device (38% a wearable)"]];
const stw = (CW - 2 * 0.4) / 3;
stats.forEach(([big, small], i) => {
  const x = X0 + i * (stw + 0.4), y = 5.2;
  hline(s, x, y, stw, MUTED, 0.75);
  T(s, big, { x, y: y + 0.08, w: stw, h: 0.55, fontFace: SERIF, fontSize: 28, color: PAPER });
  T(s, small, { x, y: y + 0.65, w: stw, h: 0.55, fontFace: SANS, fontSize: 12.5, color: MUTED_D });
});
T(s, "Before any product: non-commercial licences (ASL Citizen, NLLB-200) and the granted Sign-Speak claim need review.",
  { x: X0, y: 6.42, w: CW, h: 0.3, fontFace: SANS, fontSize: 12.5, color: MUTED_D });
notes(s, "How we compare. Against the four closest patents: Sign-Speak also uses one camera but needs a cloud server; Avodah needs several cameras and a depth sensor; Sorenson works inside its video relay service on servers; Bosch uses sensor gloves. Ours runs on one ordinary camera, fully offline, and is the only one that decides by its calibrated confidence when to answer and when to ask. The industry benchmark for accuracy is ASL Citizen: its published baseline is 63.1 percent; ours is 86.6, but on 134 words instead of 2,731, so it is not a direct comparison. Business side: about 500 to 665 dollars in parts and no per-use fees, aimed at service counters, where Deaf users in the Tran et al. survey were most willing to use such a device; 69 percent accept a stand-alone device. Honest risks: our data and translation model are non-commercial only, and Sign-Speak’s granted patent would need a legal review before any product. (About 80 seconds.)");

// ---------------------------------------------------------------- 8 references
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Comparison" });
s.addText("08 / REFERENCES", { placeholder: "label" });
s.addText("Where our facts come from", { placeholder: "title" });
const colw = (CW - 0.5) / 2, ry = 2.0;
const refBlock = (x, y, head, items, h) => {
  hline(s, x, y, colw, INK, 1.5);
  T(s, head, { x, y: y + 0.08, w: colw, h: 0.28, fontFace: MONO, fontSize: 12, bold: true, color: INK });
  T(s, items.map((t, i) => ({ text: t, options: { breakLine: i < items.length - 1, paraSpaceAfter: 7 } })),
    { x, y: y + 0.42, w: colw, h, fontFace: SANS, fontSize: 13, color: INK, lineSpacingMultiple: 1.05 });
};
refBlock(X0, ry, "PATENTS", [
  "Kelly, N. A. Realtime AI sign language recognition. US 12,518,653 B2, 2026 (Sign-Speak).",
  "Menefee, M., Nash, D., Chandler, T. Automated sign language translation using multiple input and output modalities. US 10,489,639 B2, 2019 (Avodah).",
  "Carlock, J., Maxwell, C. Translating sign language content into word content and vice versa. US 12,131,586 B2, 2024 (Sorenson).",
  "Zhou, Z. et al. System and method for automated sign language recognition. US 10,109,219 B2, 2018 (Bosch)."], 3.9);
const rx2 = X0 + colw + 0.5;
refBlock(rx2, ry, "ARTICLES", [
  "Desai, A. et al. ASL Citizen. NeurIPS 36, 2023. doi:10.52202/075280-3360",
  "Tran, N., Ladner, R. E., Bragg, D. U.S. Deaf community perspectives on automatic sign language translation. ASSETS ’23. doi:10.1145/3597638.3614507"], 1.75);
refBlock(rx2, ry + 2.3, "OTHER", [
  "MediaPipe (Lugaresi et al., 2019) · NLLB-200 (NLLB Team, 2022) · Whisper (Radford et al., 2023) · Piper TTS (Hansen, 2023) · Kaggle ISLR 1st place (Sohn, 2023) · NVIDIA Jetson Orin Nano documentation · IEEE 7000, 7002, 1012, 2841 · ANSI/UL 62368-1 · NIST AI 100-1"], 1.7);
notes(s, "Our main sources: four patents we reviewed in detail, from Sign-Speak, Avodah, Sorenson and Bosch, and two articles: the ASL Citizen dataset, which we train and test on, and the Tran et al. survey of Deaf users, which chose our use case. The other line lists the models and standards we build on; the full list of 39 references is in Section 12 of the proposal. Then: thank you, and we are happy to take one or two questions. (About 20 seconds, then questions.)");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
