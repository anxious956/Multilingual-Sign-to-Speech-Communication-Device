// Team 9 proposal presentation as .pptx (same content and look as the online deck).
// npm install pptxgenjs react react-dom react-icons sharp; then NODE_PATH=<its node_modules> PPTX_SKILL=<pptx skill dir>
// node tools/build_presentation.js . docs/presentation/njit_logo.png docs/presentation/Team9_Proposal_Presentation.pptx docs/presentation/setup_sketch.png 1078 568
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const React = require("react"), RDS = require("react-dom/server"), sharp = require("sharp");
const fa = require("react-icons/fa"), md = require("react-icons/md");
const [ROOT, LOGO, OUT, SKETCH, SK_W, SK_H] = process.argv.slice(2);
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

function box(s, x, y, w, h, title, sub, st = {}) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: st.fill || CARD },
    line: { color: st.line || INK, width: st.lw || 1, dashType: st.dash || "solid" } });
  const tc = st.tc || INK, sc = st.sc || MUTED;
  if (sub) {
    T(s, title, { x: x + 0.08, y: y + 0.06, w: w - 0.16, h: h / 2 - 0.02, valign: "bottom", align: "center", fontFace: SANS,
      fontSize: st.ts || 16, bold: true, color: tc });
    T(s, sub, { x: x + 0.08, y: y + h / 2 + 0.03, w: w - 0.16, h: h / 2 - 0.1, valign: "top", align: "center", fontFace: SANS,
      fontSize: st.ss || 12.5, color: sc });
  } else {
    T(s, title, { x: x + 0.08, y, w: w - 0.16, h, valign: "middle", align: "center", fontFace: SANS, fontSize: st.ts || 14,
      bold: true, color: tc });
  }
}
function arrow(s, x1, y1, x2, y2, o = {}) {
  const line = { color: o.color || INK, width: o.w || 1.5, endArrowType: "triangle" };
  if (o.both) line.beginArrowType = "triangle";
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1, line });
}

async function ico(C, color) {
  const svg = RDS.renderToStaticMarkup(React.createElement(C, { color: "#" + color, size: 256 }));
  return "image/png;base64," + (await sharp(Buffer.from(svg)).png().toBuffer()).toString("base64");
}
const icon = (s, data, x, y, d, name) => s.addImage({ data, x, y, w: d, h: d, altText: name });

(async () => {
const I = {
  sign: await ico(fa.FaSignLanguage, PAPER), landmarks: await ico(md.MdScatterPlot, PAPER), brain: await ico(fa.FaBrain, PAPER),
  confirm: await ico(md.MdTouchApp, INK), translate: await ico(md.MdTranslate, PAPER), speak: await ico(md.MdRecordVoiceOver, PAPER),
  chip: await ico(fa.FaMicrochip, INK), dollar: await ico(fa.FaDollarSign, INK), law: await ico(fa.FaBalanceScale, INK),
  hands: await ico(fa.FaHandsHelping, INK), clock: await ico(fa.FaClock, INK),
  cloud: await ico(fa.FaCloud, PAPER), cams: await ico(fa.FaVideo, PAPER), server: await ico(fa.FaServer, PAPER),
  glove: await ico(fa.FaHandPaper, PAPER), cam: await ico(fa.FaCamera, INK),
};
// ---------------------------------------------------------------- 1 cover
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "COVER", sectionTitle: "Introduction" });
s.addImage({ path: SKETCH, x: 7.25, y: 1.45, w: 5.2, h: 5.2 * SK_H / SK_W, altText: "Side view of the device at a service counter: the Deaf user signs to the camera and touchscreen; the speaker and microphone face the hearing staff member" });
s.addImage({ path: LOGO, x: X0, y: 0.55, w: 1.9, h: 0.5, altText: "NJIT logo" });
T(s, "ECE 414 · PROPOSAL · FALL 2026", { x: 7.5, y: 0.68, w: 4.98, h: 0.3, fontFace: MONO, fontSize: 12, charSpacing: 2,
  color: MUTED, align: "right" });
T(s, "Sign‑to‑Speech", { x: X0, y: 1.45, w: 9.5, h: 1.35, fontFace: SERIF, fontSize: 80, color: INK });
T(s, "A multilingual communication device: edge-deployed ASL recognition with confidence-gated speech output",
  { x: X0, y: 2.85, w: 5.8, h: 1.0, fontFace: SANS, fontSize: 18, color: MUTED, lineSpacingMultiple: 1.15 });
s.addShape(pres.shapes.RECTANGLE, { x: X0, y: 4.0, w: 5.8, h: 0.6, fill: { color: YEL }, line: { color: YEL } });
T(s, "Signs in. Speech out. Nothing leaves the device.", { x: X0 + 0.15, y: 4.0, w: 5.6, h: 0.6, valign: "middle",
  fontFace: SERIF, fontSize: 20, italic: true, color: INK });
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
const steps = [["1", "Sign", "one camera"], ["2", "Landmarks", "MediaPipe"], ["3", "Recognize", "+ confidence"],
               ["4", "Confirm", "touchscreen"], ["5", "Translate", "EN · TR · ES"], ["6", "Speak", "offline voice"]];
const sw = CW / 6, sy = 2.65;
hline(s, X0, sy, CW, PAPER, 1.5);
steps.forEach(([n, w, sub], i) => {
  const x = X0 + i * sw, hot = i === 3;
  if (hot) s.addShape(pres.shapes.RECTANGLE, { x, y: sy + 0.02, w: sw, h: 2.05, fill: { color: YEL }, line: { color: YEL } });
  else if (i > 0 && i !== 4) vline(s, x, sy + 0.15, 1.75, MUTED, 0.75);
  const pad = i === 0 ? 0 : 0.15;
  icon(s, I[["sign", "landmarks", "brain", "confirm", "translate", "speak"][i]], x + pad, sy + 0.22, 0.55, w);
  T(s, n, { x: x + sw - 0.45, y: sy + 0.22, w: 0.3, h: 0.3, align: "right", fontFace: MONO, fontSize: 13, color: hot ? INK : MUTED_D });
  T(s, w, { x: x + pad, y: sy + 0.9, w: sw - 0.2, h: 0.6, fontFace: SERIF, fontSize: 27, color: hot ? INK : PAPER });
  T(s, sub, { x: x + pad, y: sy + 1.52, w: sw - 0.25, h: 0.35, fontFace: SANS, fontSize: 14, color: hot ? INK : MUTED_D });
});
T(s, "The reply comes back as text  ·  nothing leaves the device", { x: X0, y: 5.45, w: CW, h: 0.45,
  fontFace: SERIF, fontSize: 22, italic: true, color: "C8C1B3" });
notes(s, "In one sentence: the device speaks for a Deaf user, but only after they say yes to the sentence. Walk the six steps left to right: the user signs to one ordinary camera; MediaPipe finds the body and hand landmarks; our recognizer reads the signs and also how sure it is; the user confirms the sentence on the touchscreen, highlighted because it is the heart of the design; NLLB-200 translates it into English, Turkish or Spanish; and Piper speaks it. The hearing person’s reply comes back through Whisper as English text on the same screen. Everything runs offline on an NVIDIA Jetson, so no video ever leaves the device. It is an aid for when no interpreter is available, not a replacement. (About 60 seconds.)");

// ---------------------------------------------------------------- 3 specs
pres.addSection({ title: "Requirements" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Requirements" });
s.addText("03 / DESIGN OBJECTIVES & SPECIFICATIONS", { placeholder: "label" });
s.addText("Four numbers that matter most", { placeholder: "title" });
const big4 = [["86.6%", "Top-1 accuracy", "target \u2265 75%"], ["98.2%", "Top-5 accuracy", "target \u2265 90%"],
              ["0.7 s", "Confirm \u2192 speech", "target < 1.2 s"], ["4.4 GB", "All models", "limit 8 GB"]];
const bw4 = 2.6;
big4.forEach(([n, what, tgt], i) => {
  const x = X0 + (i % 2) * (bw4 + 0.35), y = 2.05 + Math.floor(i / 2) * 1.85;
  hline(s, x, y, bw4, INK, 1.5);
  T(s, n, { x, y: y + 0.12, w: bw4, h: 0.8, fontFace: SERIF, fontSize: 44, color: BLUE });
  T(s, what, { x, y: y + 0.95, w: bw4, h: 0.35, fontFace: SANS, fontSize: 16, bold: true, color: INK });
  T(s, tgt, { x, y: y + 1.32, w: bw4, h: 0.3, fontFace: MONO, fontSize: 13, color: MUTED });
});
s.addShape(pres.shapes.RECTANGLE, { x: X0, y: 5.95, w: 0.9, h: 0.36, fill: { color: YEL }, line: { color: YEL } });
T(s, "NEXT", { x: X0, y: 5.95, w: 0.9, h: 0.36, align: "center", valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: INK });
T(s, "same tests on the Jetson and our own camera", { x: X0 + 1.05, y: 5.95, w: 4.6, h: 0.36, valign: "middle",
  fontFace: SANS, fontSize: 14, color: INK });
const gx = 6.95, gw = X0 + CW - gx;
s.addChart(pres.charts.BAR, [
  { name: "Top-1", labels: ["v1", "v2", "v3", "v4", "v5"], values: [61.0, 62.3, 66.4, 67.2, 86.6] },
  { name: "Top-5", labels: ["v1", "v2", "v3", "v4", "v5"], values: [77.6, 79.5, 83.0, 83.9, 98.2] }],
  { x: gx, y: 1.95, w: gw, h: 4.3, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [BLUE, MUTED_D], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: INK,
    dataLabelFormatCode: "0.0", dataLabelFontFace: "+mn-lt",
    valAxisMinVal: 40, valAxisMaxVal: 100, valAxisMajorUnit: 20, valAxisLabelColor: MUTED, valAxisLabelFontSize: 10,
    valAxisLabelFontFace: "+mn-lt", catAxisLabelColor: INK, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: "D6CFC0", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendFontSize: 11, legendColor: INK, legendFontFace: "+mn-lt",
    showTitle: true, title: "Our recognizer, version by version (% on unseen signers)", titleFontSize: 13, titleColor: INK,
    titleFontFace: "+mn-lt", objectName: "accuracy chart" });
T(s, "v1\u2013v4: Kaggle data, 250 signs \u00b7 v5: ASL Citizen, 134 words", { x: gx, y: 6.3, w: gw, h: 0.3,
  fontFace: MONO, fontSize: 11, color: MUTED });
notes(s, "Our design objectives as measurable specifications; these four matter most. Recognition must reach at least 75 percent top-1 and 90 percent top-5 on signers the model has never seen; our v5 model reaches 86.6 and 98.2 percent on the 11 Deaf test signers of the ASL Citizen dataset. Speech must start within 1.2 seconds of the user’s confirmation; translation plus the first audio takes about 0.7 seconds on an ordinary CPU. All models together are estimated at 4.4 of the Jetson’s 8 gigabytes. The other requirements: a vocabulary of 100 to 250 signs (our draft has 134 words), three spoken languages, no network, and at least 15 frames per second (about 20 on a laptop so far). Next we repeat these tests on the Jetson and with our own camera. If asked: the 86.6 percent is on recorded videos, not yet live. (About 75 seconds.)");

// ---------------------------------------------------------------- 4 constraints
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Requirements" });
s.addText("04 / CONSTRAINTS & STANDARDS", { placeholder: "label" });
s.addText("A small box, a student budget, strict rules", { placeholder: "title" });
const cons = [["Technical", "8 GB · 1.2 s · one camera · offline"], ["Economic", "≈ $500, paid by the team"],
  ["Legal", "ADA · non-commercial licences · IRB"], ["Ethical", "Confirm before speaking"],
  ["Time, safety", "ECE 414 → ECE 416 · heat, power"]];
const cy0 = 2.05, ch = 0.86, cw = 6.85;
cons.forEach(([k, v], i) => {
  const y = cy0 + i * ch;
  hline(s, X0, y, cw, i === 0 ? INK : RULE, i === 0 ? 1.5 : 0.75);
  icon(s, I[["chip", "dollar", "law", "hands", "clock"][i]], X0 + 0.02, y + 0.22, 0.4, k);
  T(s, k, { x: X0 + 0.6, y: y + 0.2, w: 1.9, h: 0.45, fontFace: SERIF, fontSize: 22, color: INK });
  T(s, v, { x: X0 + 2.6, y: y + 0.24, w: cw - 2.6, h: 0.45, fontFace: SANS, fontSize: 17, color: "3D3A35" });
});
hline(s, X0, cy0 + 5 * ch, cw, INK, 1.5);
const px = 8.15, pw = X0 + CW - px;
s.addShape(pres.shapes.RECTANGLE, { x: px, y: cy0, w: pw, h: 5 * ch, fill: { color: INK }, line: { color: INK } });
T(s, "STANDARDS", { x: px + 0.3, y: cy0 + 0.25, w: pw - 0.6, h: 0.3, fontFace: MONO, fontSize: 13, charSpacing: 2, color: YEL });
const stds = [["IEEE 7000", "ethical design"], ["IEEE 7002", "privacy"], ["IEEE 1012", "verification, validation"],
  ["IEEE 2841", "AI model evaluation"], ["UL 62368-1", "equipment safety"], ["NIST AI RMF", "AI risk"]];
stds.forEach(([c, d], i) => {
  const y = cy0 + 0.72 + i * 0.6;
  hline(s, px + 0.3, y, pw - 0.6, MUTED, 0.75);
  T(s, c, { x: px + 0.3, y: y + 0.14, w: 1.75, h: 0.32, fontFace: MONO, fontSize: 14, bold: true, color: PAPER });
  T(s, d, { x: px + 2.1, y: y + 0.14, w: pw - 2.4, h: 0.32, fontFace: SANS, fontSize: 14, color: MUTED_D });
});
notes(s, "Our constraints in five groups. Technical: everything must fit in the Jetson’s 8 gigabytes and meet the 1.2 second budget with one ordinary camera and no network. Economic: about 500 to 665 dollars in parts, paid by the four of us, so only free, open-source models. Legal: under the ADA the device must never be presented as a replacement for an interpreter; the ASL Citizen dataset and the NLLB translation model are licensed for non-commercial use only; recordings of people need written consent and an IRB check. Ethical: nothing is spoken without the user’s confirmation, and accuracy is reported for each signer. Time and safety: we propose now and build in ECE 416, keeping heat and power in check. On the right, the six standards we follow: IEEE 7000 for ethical design, 7002 for privacy, 1012 for verification and validation, 2841 for evaluating deep-learning models, UL 62368-1 for electrical and thermal safety, and the NIST AI risk management framework. (About 75 seconds.)");

// ---------------------------------------------------------------- 5 block diagram
pres.addSection({ title: "Design" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Design" });
s.addText("05 / COMPONENTS & BLOCK DIAGRAM", { placeholder: "label" });
s.addText("One $250 board does all the thinking", { placeholder: "title" });
const colL = X0, colW = 2.55, cx0 = 4.55, cW = 4.25, colR = X0 + CW - colW;
const lab = (t, x, w) => T(s, t, { x, y: 1.88, w, h: 0.3, align: "center", fontFace: MONO, fontSize: 11, charSpacing: 1, color: MUTED });
lab("FACES THE DEAF USER", colL, colW); lab("INSIDE THE BASE", cx0, cW); lab("FACES THE HEARING PERSON", colR, colW);
const r5 = [2.3, 3.33, 4.36], bh5 = 0.85, mid = (i) => r5[i] + bh5 / 2;
box(s, colL, r5[0], colW, bh5, "Camera", "1080p, USB");
box(s, colL, r5[1], colW, bh5, "Touchscreen", "confirm, read reply");
box(s, colL, r5[2], colW, bh5, "Button + LED", "LED on while recording");
box(s, colR, r5[0], colW, bh5, "Speaker", "speaks the sentence");
box(s, colR, r5[1], colW, bh5, "Microphone", "records the reply");
box(s, colR, r5[2], colW, bh5, "Wi-Fi card", "removed: fully offline", { dash: "dash", line: MUTED, tc: MUTED });
const jy = r5[0], jh = r5[2] + bh5 - r5[0];
s.addShape(pres.shapes.RECTANGLE, { x: cx0, y: jy, w: cW, h: jh, fill: { color: INK }, line: { color: INK } });
T(s, "Jetson Orin Nano", { x: cx0 + 0.3, y: jy + 0.28, w: cW - 0.6, h: 0.6, fontFace: SERIF, fontSize: 28, color: PAPER });
T(s, "6-core Arm CPU · 1024-core GPU · 8 GB memory", { x: cx0 + 0.3, y: jy + 1.0, w: cW - 0.6, h: 0.6,
  fontFace: SANS, fontSize: 14, color: "C8C1B3" });
T(s, "Runs every model, with no network", { x: cx0 + 0.3, y: jy + 1.7, w: cW - 0.6, h: 0.35,
  fontFace: SANS, fontSize: 14, color: "C8C1B3" });
s.addShape(pres.shapes.RECTANGLE, { x: cx0 + cW - 1.25, y: jy + jh - 0.6, w: 0.95, h: 0.36, fill: { color: YEL }, line: { color: YEL } });
T(s, "$250", { x: cx0 + cW - 1.25, y: jy + jh - 0.6, w: 0.95, h: 0.36, align: "center", valign: "middle", fontFace: MONO,
  fontSize: 13, bold: true, color: INK });
const by5 = r5[2] + bh5 + 0.38, bb = 0.72;
box(s, cx0, by5, 1.95, bb, "NVMe SSD", "OS and models");
box(s, cx0 + cW - 1.95, by5, 1.95, bb, "Power", "19 V or battery");
arrow(s, colL + colW, mid(0), cx0, mid(0));
arrow(s, cx0, mid(1), colL + colW, mid(1), { both: true });
arrow(s, colL + colW, mid(2), cx0, mid(2), { both: true });
arrow(s, cx0 + cW, mid(0), colR, mid(0));
arrow(s, colR, mid(1), cx0 + cW, mid(1));
arrow(s, cx0 + 0.97, by5, cx0 + 0.97, jy + jh, { both: true });
arrow(s, cx0 + cW - 0.97, by5, cx0 + cW - 0.97, jy + jh);
const tag = (t, x, y, w) => T(s, t, { x, y, w, h: 0.26, align: "center", fontFace: MONO, fontSize: 11, color: BLUE });
const gl = colL + colW, GAPL = cx0 - gl, gr = cx0 + cW, GAPR = colR - gr;
tag("USB", gl, mid(0) - 0.32, GAPL); tag("HDMI, USB", gl, mid(1) - 0.32, GAPL); tag("GPIO", gl, mid(2) - 0.32, GAPL);
tag("USB", gr, mid(0) - 0.32, GAPR); tag("USB", gr, mid(1) - 0.32, GAPR);
tag("PCIe", cx0 + 1.02, jy + jh + 0.06, 0.7); tag("19 V", cx0 + cW - 0.92, jy + jh + 0.06, 0.7);
T(s, "PARTS ≈ $495–665", { x: colR, y: by5 + 0.2, w: colW, h: 0.32, align: "center", fontFace: MONO, fontSize: 13,
  bold: true, color: INK });
notes(s, "This is the hardware block diagram. Everything that thinks is one NVIDIA Jetson Orin Nano developer kit in the base, about 250 dollars. On the Deaf user’s side: the camera, the touchscreen, and a button and LED; the LED lights whenever the camera or microphone is on. On the hearing person’s side: speaker and microphone, all on USB. The board has only DisplayPort, so the touchscreen uses an adapter. We remove the Wi-Fi card and leave Ethernet unplugged, so offline is a physical fact you can check. Parts for one prototype total about 495 to 665 dollars. On the module run MediaPipe, our 1.78-million-parameter recognizer, which needs about 4 milliseconds per sign, NLLB-200 for translation, Piper for speech and Whisper for the reply. (About 75 seconds.)");

// ---------------------------------------------------------------- 6 prototyping
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Design" });
s.addText("06 / PROTOTYPING, VALIDATION, TESTING & ASSEMBLY", { placeholder: "label" });
s.addText("Cheap prototypes first, then the real device", { placeholder: "title" });
const fy = 1.95;
const mx = X0, mw = 3.9, sxx = mx + mw + 0.4, sw6 = 1.6, rr = 0.6;
const ry6 = [2.0, 2.85, 3.75, 4.85, 5.65];
box(s, mx, ry6[0], mw, rr, "Objectives & specifications", null, { ts: 14 });
box(s, sxx, ry6[0], sw6, rr, "Benchmarking", null, { fill: BAND, ts: 13 });
box(s, mx, ry6[1], mw, rr, "Concept & constraints", null, { ts: 14 });
box(s, sxx, ry6[1], sw6, rr, "Business", null, { fill: BAND, ts: 13 });
arrow(s, mx + mw, ry6[0] + rr / 2, sxx, ry6[0] + rr / 2, { both: true });
arrow(s, mx + mw, ry6[1] + rr / 2, sxx, ry6[1] + rr / 2, { both: true });
arrow(s, mx + mw / 2, ry6[0] + rr, mx + mw / 2, ry6[1]);
const cw6 = (mw - 2 * 0.12) / 3, ch6 = 0.8;
["Recognition", "Language, speech", "Embedded"].forEach((t, i) => {
  const x = mx + i * (cw6 + 0.12), c = x + cw6 / 2;
  box(s, x, ry6[2], cw6, ch6, t, null, { fill: INK, line: INK, tc: PAPER, ts: 12 });
  arrow(s, c, ry6[1] + rr, c, ry6[2], { both: true });
  arrow(s, c, ry6[2] + ch6, c, ry6[3]);
});
box(s, mx, ry6[3], mw, rr, "Prototyping, testing, integration", null, { fill: YEL, line: INK, ts: 14 });
arrow(s, mx + mw / 2, ry6[3] + rr, mx + mw / 2, ry6[4]);
box(s, mx, ry6[4], mw, rr, "Prototype assembly", null, { fill: YEL, line: INK, ts: 14 });
T(s, "COURSE DESIGN FLOW · OUR THREE GOALS IN BLACK", { x: mx, y: 6.33, w: 6, h: 0.28, fontFace: MONO, fontSize: 11, color: MUTED });
const rx = 7.05, rw = X0 + CW - rx, rh = 0.52;
const protos = [["P1", "Laptop demo", "BUILT"], ["P2", "Paper mock-up", "NEXT"], ["P3", "Jetson on the bench", "SOON"],
  ["P4", "3D-printed case", "PLANNED"], ["P5", "Full device", "ECE 416"]];
protos.forEach(([c, t, st], i) => {
  const y = fy + i * rh;
  hline(s, rx, y, rw, i === 0 ? INK : RULE, i === 0 ? 1.5 : 0.75);
  T(s, c, { x: rx, y: y + 0.12, w: 0.55, h: 0.32, fontFace: MONO, fontSize: 14, bold: true, color: INK });
  T(s, t, { x: rx + 0.6, y: y + 0.11, w: rw - 2.1, h: 0.34, fontFace: SANS, fontSize: 17, color: INK });
  if (i === 0) s.addShape(pres.shapes.RECTANGLE, { x: rx + rw - 1.4, y: y + 0.1, w: 1.4, h: 0.32, fill: { color: YEL }, line: { color: YEL } });
  T(s, st, { x: rx + rw - 1.4, y: y + 0.1, w: 1.4, h: 0.32, valign: "middle", align: "center", fontFace: MONO, fontSize: 13,
    color: i === 0 ? INK : BLUE });
});
hline(s, rx, fy + 5 * rh, rw, INK, 1.5);
T(s, "85%", { x: rx, y: fy + 5 * rh + 0.2, w: 1.9, h: 0.85, fontFace: SERIF, fontSize: 50, color: BLUE });
T(s, "of signs answered at 6.5% error; the user is asked about the rest", { x: rx + 2.0, y: fy + 5 * rh + 0.3, w: rw - 2.0,
  h: 0.7, fontFace: SANS, fontSize: 15, color: INK });
T(s, "UNIT → INTEGRATION → SYSTEM → DEAF USERS", { x: rx, y: fy + 5 * rh + 1.25, w: rw, h: 0.35,
  fontFace: MONO, fontSize: 13, color: "3D3A35" });
notes(s, "On the left is the course’s design flow chart, filled in for our project: our three goals are components X, Y and Z, and we added one loop back to the concept when a test misses its target. We prototype from cheap to expensive. P1, the proof of principle, is built: a laptop and webcam running the full recognizer live. P2 tests the confirmation screen with Deaf users on paper, with a teammate playing the computer, the Wizard of Oz method. P3 is the Jetson on the bench, P4 the 3D-printed enclosure, and P5 the assembled device in ECE 416. Validation so far is on recorded data from 11 unseen Deaf signers: at our confidence threshold the device answers 85 percent of signs at 6.5 percent error and asks the user about the rest. Testing goes from unit to integration to 15 system requirements, ending with Deaf users at a mock service counter. (About 80 seconds.)");

// ---------------------------------------------------------------- 7 benchmarking
pres.addSection({ title: "Comparison" });
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Comparison" });
s.addText("07 / BENCHMARKING · BUSINESS & ECONOMIC POTENTIAL", { placeholder: "label" });
s.addText([{ text: "Others need a cloud, gloves or depth cameras. " }, { text: "We need one webcam", options: { italic: true, color: YEL } }],
  { placeholder: "title" });
const kx = X0, kw = 6.9, ky = 3.1, kh = 0.56;
const comp = [["Sign-Speak", "a cloud server"], ["Avodah", "cameras + depth sensor"], ["Sorenson", "relay servers"],
              ["Bosch", "sensor gloves"], ["Our device", "one webcam, offline"]];
T(s, "PATENTED SYSTEMS NEED", { x: kx, y: ky - 0.4, w: kw, h: 0.3, fontFace: MONO, fontSize: 12, charSpacing: 2, color: MUTED_D });
comp.forEach(([sys, need], i) => {
  const y = ky + i * kh, ours = i === 4;
  if (ours) s.addShape(pres.shapes.RECTANGLE, { x: kx, y, w: kw, h: kh, fill: { color: YEL }, line: { color: YEL } });
  else hline(s, kx, y, kw, i === 0 ? PAPER : MUTED, i === 0 ? 1.5 : 0.75);
  icon(s, I[["cloud", "cams", "server", "glove", "cam"][i]], kx + 0.12, y + 0.1, 0.36, sys);
  T(s, sys, { x: kx + 0.65, y, w: 2.0, h: kh, valign: "middle", fontFace: SANS, fontSize: 17, bold: true, color: ours ? INK : PAPER });
  T(s, need, { x: kx + 2.6, y, w: kw - 2.7, h: kh, valign: "middle", fontFace: SANS, fontSize: 17, color: ours ? INK : "C8C1B3" });
});
const sx = 8.3, sw2 = X0 + CW - sx;
const stats = [["63.1% → 86.6%", "accuracy: published baseline vs ours*"], ["$495–665", "parts, no per-use fees"],
               ["69%", "of Deaf users accept a stand-alone device"]];
stats.forEach(([big, small], i) => {
  const y = ky - 0.4 + i * 1.3;
  hline(s, sx, y, sw2, MUTED, 0.75);
  T(s, big, { x: sx, y: y + 0.1, w: sw2, h: 0.6, fontFace: SERIF, fontSize: 30, color: PAPER });
  T(s, small, { x: sx, y: y + 0.72, w: sw2, h: 0.4, fontFace: SANS, fontSize: 14, color: MUTED_D });
});
T(s, "* ASL Citizen: 2,731 signs vs our 134-word subset", { x: sx, y: 6.42, w: sw2, h: 0.3, fontFace: SANS, fontSize: 11, color: MUTED_D });
notes(s, "How we compare. The four closest patents each need something extra: Sign-Speak, granted this year, uses one camera but needs a cloud server; Avodah needs several cameras and a depth sensor; Sorenson works on servers inside its video relay service; Bosch uses sensor gloves. Ours needs one ordinary webcam and runs fully offline, and it is the only one that decides by calibrated confidence when to answer and when to ask the user. The industry benchmark for accuracy is the ASL Citizen dataset: its published baseline is 63.1 percent; ours is 86.6, but on 134 words instead of 2,731, so it is not a direct comparison. Business side: about 500 to 665 dollars in parts and no per-use or cloud fees, aimed at service counters and front desks, where Deaf users in the Tran et al. survey were most willing to use such a device; 69 percent accept a stand-alone device, only 38 percent a wearable. Honest risks: our dataset and translation model are non-commercial only, and Sign-Speak’s granted patent would need a legal review before any product. (About 80 seconds.)");

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
  "Zhou, Z. et al. System and method for automated sign language recognition. US 10,109,219 B2, 2018 (Bosch)."], 2.3);
const rx2 = X0 + colw + 0.5;
refBlock(rx2, ry, "ARTICLES", [
  "Desai, A. et al. ASL Citizen. NeurIPS 36, 2023. doi:10.52202/075280-3360",
  "Tran, N., Ladner, R. E., Bragg, D. U.S. Deaf community perspectives on automatic sign language translation. ASSETS ’23. doi:10.1145/3597638.3614507"], 1.1);
refBlock(rx2, ry + 2.15, "OTHER", [
  "MediaPipe (Lugaresi et al., 2019) · NLLB-200 (NLLB Team, 2022) · Whisper (Radford et al., 2023) · Piper TTS (Hansen, 2023) · Kaggle ISLR 1st place (Sohn, 2023) · NVIDIA Jetson Orin Nano documentation · IEEE 7000, 7002, 1012, 2841 · ANSI/UL 62368-1 · NIST AI 100-1"], 1.0);
// grey logos under each group (Sorenson's logo could not be obtained; its name is in the list)
const LOGO_DIR = path.join(ROOT, "docs", "presentation", "logos");
const LOGOS = JSON.parse(require("fs").readFileSync(path.join(LOGO_DIR, "sizes.json"), "utf8"));
const logoRow = (keys, x, y, h, gap) => {
  let cx = x;
  keys.forEach((k) => {
    const [w0, h0] = LOGOS[k], w = h * w0 / h0;
    s.addImage({ path: path.join(LOGO_DIR, k + ".png"), x: cx, y, w, h, altText: k + " logo" });
    cx += w + gap;
  });
};
logoRow(["signspeak", "avodah", "bosch"], X0, 5.1, 0.32, 0.4);
logoRow(["neurips", "acm"], rx2, 3.6, 0.42, 0.4);
logoRow(["google", "meta", "openai"], rx2, 5.68, 0.24, 0.3);
logoRow(["nvidia", "ieee", "nist"], rx2, 6.08, 0.24, 0.3);
notes(s, "Our main sources: four patents we reviewed in detail, from Sign-Speak, Avodah, Sorenson and Bosch, and two articles: the ASL Citizen dataset, which we train and test on, and the Tran et al. survey of Deaf users, which chose our use case. The other line lists the models and standards we build on; the full list of 39 references is in Section 12 of the proposal. Then: thank you, and we are happy to take one or two questions. (About 20 seconds, then questions.)");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
