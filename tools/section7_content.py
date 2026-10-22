"""Text, tables and references of proposal Section 7, used by tools/add_section7_to_proposal.py.

Markup: **bold**, *italic*, {cite:key} for a new reference (numbered from 30), {ref:N} for an existing one [N].
Measured numbers are read from docs/figures/section7_benchmarks.json (tools/bench_section7.py).
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")
FIRST_FIG, FIRST_TAB = 5, 9
B = json.load(open(os.path.join(FIG, "section7_benchmarks.json")))

REFERENCES = {
    "nvidia_specs": "NVIDIA. Jetson Orin Nano Super Developer Kit: technical specifications. *NVIDIA*, 2026. "
                    "Available: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/"
                    "nano-super-developer-kit/ [Accessed: 22-Oct-2026].",
    "hyman": "Hyman, B. *Fundamentals of Engineering Design*, 2nd ed. Prentice Hall, Upper Saddle River, NJ, 2003.",
    "guo": "Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. On calibration of modern neural networks. "
           "*Proceedings of the 34th International Conference on Machine Learning (ICML), PMLR*, 70, 1321–1330, 2017.",
    "onnxruntime": "ONNX Runtime developers. ONNX Runtime: cross-platform, high performance ML inferencing and "
                   "training accelerator. *GitHub repository*, 2026. Available: https://github.com/microsoft/onnxruntime "
                   "[Accessed: 22-Oct-2026].",
    "nvidia_guide": "NVIDIA. Jetson Orin Nano Developer Kit user guide: hardware layout and quick start. "
                    "*NVIDIA Developer Documentation*, 2026. Available: "
                    "https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/ [Accessed: 22-Oct-2026].",
    "ctranslate2": "OpenNMT. CTranslate2: fast inference engine for Transformer models. *GitHub repository*, 2026. "
                   "Available: https://github.com/OpenNMT/CTranslate2 [Accessed: 22-Oct-2026].",
    "wiki_proto": "Wikipedia contributors. Prototype. *Wikipedia, The Free Encyclopedia*, 2026. Available: "
                  "https://en.wikipedia.org/wiki/Prototype [Accessed: 22-Oct-2026].",
    "wiki_vv": "Wikipedia contributors. Verification and validation. *Wikipedia, The Free Encyclopedia*, 2026. "
               "Available: https://en.wikipedia.org/wiki/Verification_and_validation [Accessed: 22-Oct-2026].",
    "nasa": "NASA. *NASA Systems Engineering Handbook*. NASA/SP-2016-6105 Rev2, National Aeronautics and Space "
            "Administration, Washington, DC, 2016.",
}


def cpu_label(name):
    """'Intel(R) Xeon(R) Processor @ 2.80GHz' -> 'Intel Xeon, 2.8 GHz'."""
    name = re.sub(r"\((R|TM)\)|\bProcessor\b|\bCPU\b", "", name)
    m = re.search(r"@\s*([\d.]+)\s*GHz", name)
    base = " ".join(name.split("@")[0].split())
    return f"{base}, {float(m.group(1)):g} GHz" if m else base


def f1(x):
    return f"{x:.1f}"


def build(b):
    r, n, p, w = B["recognizer"], B["nllb"], B["piper"], B["whisper"]
    sim = B["simulation"]

    b.page_break()
    b.heading(1, "7. Design Methodology, Prototyping, Validation and Hardware Block Diagram")
    b.body(
        "This section describes how the device is designed, built and tested. The design follows the flow chart "
        "recommended for ECE capstone projects, adapted to the three technical goals of Section 4 (Fig. 5). The "
        "flow chart is explained block by block in Sections 7.2 and 7.3, and the hardware it leads to is shown as a "
        "block diagram in Section 7.4 (Fig. 6). Section 7.5 gives the preliminary data and simulations available "
        "so far. Section 7.6 describes the five prototypes we will build and the category each belongs to, Section "
        "7.7 describes how they will be verified and validated, and Section 7.8 describes how the final prototype "
        "will be assembled.")

    # ---------------------------------------------------------------- 7.1
    b.heading(2, "7.1 Design Flow")
    b.body(
        "Fig. 5 shows the design flow of the project. It keeps the blocks and links of the recommended chart and "
        "replaces the generic components X, Y and Z with our three technical goals. Design starts from the "
        "objectives, requirements and specifications of Tables 1 and 3, which are compared with published results, "
        "patented systems and the specifications of the target hardware {cite:nvidia_specs} (benchmarking). They "
        "lead to a design concept that respects the constraints of Section 6.2 and is checked against business and "
        "economic factors. The concept is divided into three components, which the responsible team members design "
        "in parallel (Table 4) and which meet again in prototyping, testing and integration. Double arrows mark "
        "steps that feed back into each other: a component that cannot meet its target on the embedded hardware "
        "changes the concept, and a benchmark that turns out to be easy to beat changes the specification. We added "
        "one link to the recommended chart, the dashed line on the left: when a test misses its target, the work "
        "returns to the design concept instead of moving on to assembly.")
    b.body(
        "The order of the work is driven by risk. The highest risks in Table 8 are tested first, with the cheapest "
        "prototype that can answer the question. Sign recognition was therefore prototyped first, on a laptop, "
        "and landmark extraction on the Jetson will be the first test once the hardware arrives.")
    b.figure(os.path.join(FIG, "fig_design_flow.png"), 6.3,
             "Design flow chart of the project, based on the flow chart recommended for ECE capstone senior design "
             "projects. Components X, Y and Z are the three technical goals of Section 4, and the italic line at the "
             "bottom of each gives its status in October 2026. The dashed feedback line was added to the "
             "recommended chart.")

    # ---------------------------------------------------------------- 7.2
    b.page_break()
    b.heading(2, "7.2 Design Objectives, Benchmarking and Design Concept")
    b.body(
        "**Design objectives, requirements and specifications.** The objectives come from the problem statement "
        "in Section 3.2 and from the needs Deaf users reported in surveys {ref:1}, {ref:16}. They are stated as "
        "measurable targets in Tables 1 and 3 and as constraints in Section 6.2. In Section 7.7 every requirement "
        "receives an identifier, a verification method and a pass criterion (Table 16), so that each specification "
        "can be traced to the test that checks it, as IEEE 1012 requires {ref:26}.")
    b.body(
        "**Benchmarking.** Table 9 lists the benchmarks the design is compared with. They show which targets are "
        "realistic and where the design must differ from existing systems. The ASL Citizen baseline {ref:2} set "
        "the accuracy target: 75% top-1 accuracy is above the best published result on unseen signers, but for a "
        "much smaller vocabulary than the 2,731 signs of the full dataset. The Kaggle first-place solution {ref:7} "
        "became the starting architecture of our recognizer. The patented systems of Table 2 show what the design "
        "must not copy, and the specifications of the Jetson Orin Nano set the memory and power budget for "
        "everything else.")
    b.table("Benchmarks used in the design and the current position of the project.",
            [1900, 2500, 1960, 3000],
            ["Benchmark", "What it represents", "Reference value", "Our design or result"],
            [["ASL Citizen baseline {ref:2}", "Best published isolated-sign accuracy on signers not seen in training",
              "63.1% top-1 on all 2,731 signs",
              f"v5: {f1(r['v5_top1'])}% top-1 on a 134-word subset (not directly comparable: smaller vocabulary)"],
             ["Kaggle first-place solution {ref:7}", "State-of-the-art landmark model and training recipe",
              "1D-CNN + transformer on 250 signs", "Recipe adopted in v3: +4.1 points top-1 over our v2 on the same "
              "test signers"],
             ["Sign-Speak patent {ref:19}", "Closest existing system: one camera, menu of three candidates",
              "Recognition on a remote server", "Same candidate idea with five candidates, fully offline"],
             ["SignAll {ref:17}, Avodah {ref:18}", "Commercial and patented multi-sensor systems",
              "Several cameras or a depth sensor", "One ordinary RGB camera"],
             ["Jetson Orin Nano {cite:nvidia_specs}", "Capacity of the target platform",
              "8 GB shared memory, 1024 CUDA cores, 7–25 W", f"All models estimated at {f1(sim['mem_total_gb'])} GB "
              "(Fig. 9(c))"]])
    b.body(
        "**Design concept.** The concept is the chain shown in Fig. 2: one RGB camera, landmarks instead of video, "
        "recognition of single signs with a calibrated confidence, a confirmation screen, and offline translation "
        "and speech. Each link of the chain was chosen from several alternatives. Table 10 lists the main "
        "decisions, the options considered and the reason for each choice, with the evidence we have so far.")
    b.table("Main design decisions and the alternatives considered.",
            [1500, 2350, 2110, 3400],
            ["Decision", "Options considered", "Chosen", "Reason and evidence"],
            [["Input sensing", "Sensor gloves; depth camera; several cameras; one RGB camera", "One RGB camera",
              "Nothing to wear and lowest cost; 69% of surveyed Deaf users were comfortable with a stand-alone "
              "system, 38% with wearables {ref:1}"],
             ["Model input", "Video frames; body and hand landmarks", "94 of the 543 MediaPipe landmarks {ref:6}",
              f"Small and fast model ({f1(r['params_M'] * 1.0)} M parameters, {f1(r['ort_ms_1t'])} ms per sign on "
              "one CPU core); the Kaggle training data shrank from 56 GB to 2.02 GB"],
             ["Recognition task", "Continuous translation; isolated signs from a fixed vocabulary",
              "Isolated signs, rest pose between signs",
              "Continuous translation still needs large models and data {ref:4}, {ref:5}; isolated signs can be "
              "tested one by one and give a confidence for each"],
             ["Handling uncertainty", "Always speak the top sign; abstain and confirm",
              "Calibrated threshold, five candidates, confirmation before speech",
              f"Selective prediction {{ref:14}}; v5 accepts {f1(B['calib']['coverage'])}% of signs at "
              f"{f1(B['calib']['error'])}% error, and the right sign is among five candidates "
              f"{f1(r['v5_top5'])}% of the time"],
             ["Where processing runs", "Cloud server; on the device", "On the device, fully offline",
              "No video leaves the device and no network is needed; the main difference from Sign-Speak {ref:19}"],
             ["Processing platform", "Jetson Orin Nano; Raspberry Pi 5; laptop; cloud", "Jetson Orin Nano",
              "Highest score in the weighted decision matrix (Table 11)"],
             ["Signs to sentence", "Rules only; large language model; templates with a small model as fallback",
              "Templates + small language model",
              "Templates are fast and predictable for frequent phrases; the model covers the rest; every sentence "
              "is confirmed by the user"],
             ["Translation and speech", "Cloud services; offline open models", "NLLB-200, Piper, Whisper",
              "Run offline in all three languages; first timings in Table 13"]],
            bold_first_col=True)
    b.body(
        "For the processing platform we used a weighted decision matrix, a method described in the course "
        "textbook {cite:hyman}. Each option was scored from 1 (poor) to 5 (excellent) on six criteria, and the "
        "scores were multiplied by the weight of each criterion (Table 11). The weights follow the requirements: "
        "running all models in real time matters most, followed by privacy and offline operation. The scores are "
        "the team's judgment, based on published specifications and our own measurements on a laptop and a cloud "
        "processor (Section 7.5). The Jetson Orin Nano scores highest because it is the only option that combines "
        "a graphics processor for the larger models with a small, low-power module. A Raspberry Pi 5 would be "
        "cheaper and smaller, but without a usable graphics processor the translation and speech recognition models "
        "would leave little time for the rest of the pipeline. A laptop is fast but costs more, is larger, and is "
        "not a realistic basis for a product. A cloud server fails the offline constraint and is shown only for "
        "comparison.")
    crit = [("Runs all models in real time", 0.30, [4, 2, 5, 5]),
            ("Privacy and offline operation", 0.20, [5, 5, 5, 1]),
            ("Cost", 0.15, [4, 5, 2, 3]),
            ("Size and power", 0.15, [4, 5, 2, 3]),
            ("Software support (CUDA, TensorRT, ONNX)", 0.10, [5, 3, 5, 4]),
            ("Path to a product", 0.10, [4, 4, 2, 2])]
    totals = [sum(c[1] * c[2][i] for c in crit) for i in range(4)]
    rows = [[c[0], f"{c[1]:.2f}"] + [str(s) for s in c[2]] for c in crit]
    rows.append(["Weighted score (out of 5)", "1.00"] + [f"{t:.2f}" for t in totals])
    b.table("Weighted decision matrix for the processing platform (scores from 1 = poor to 5 = excellent).",
            [3060, 900, 1350, 1350, 1350, 1350],
            ["Criterion", "Weight", "Jetson Orin Nano", "Raspberry Pi 5", "Laptop with GPU", "Cloud server*"],
            rows, center_cols=(1, 2, 3, 4, 5), total_row=True)
    b.body("*Fails the offline requirement; shown for comparison only.", indent=False)
    b.body(
        "**Business and economic perspectives.** The prototype costs about $485–$640 in parts (Table 6), uses "
        "open-source software, and has no per-use or cloud fees, which keeps its cost far below that of repeated "
        "interpreter services {ref:1}. Two economic factors shaped the design. First, the parts are standard and "
        "replaceable, so a failed part does not end the device's life. Second, ASL Citizen and NLLB-200 may be used "
        "only for non-commercial purposes (Section 6.2). Translation is therefore kept behind a narrow interface, "
        "an English sentence in and a translated sentence out, so that a commercially licensed model could later "
        "replace NLLB-200 without changes elsewhere. A product would also use the Jetson Orin Nano module on a "
        "smaller custom carrier board instead of the developer kit.")

    # ---------------------------------------------------------------- 7.3
    b.heading(2, "7.3 Component Design")
    b.body(
        "The concept is divided into three components, one per technical goal (Fig. 5). They exchange data through "
        "interfaces agreed at the start of the project, so that each can be developed and tested with saved sample "
        "data before the others are ready. The recognizer writes one line of JavaScript Object Notation (JSON) per "
        "stretch of signing, with the five most likely signs, their probabilities, their start and end times and "
        "an abstain flag, and the sentence builder reads these lines. The confirmed English sentence and the chosen "
        "language are passed on to translation and speech. Agreeing on these interfaces first lets the three goals "
        "progress in parallel and makes the integration tests of Section 7.7 simple to define.")
    b.body(
        "**Component X: sign recognition (Goal 1).** MediaPipe Holistic {ref:6} converts each camera frame into "
        "body, hand and face landmarks. Of its 543 landmarks the recognizer keeps 94: 40 around the lips, 21 on each "
        "hand and 12 on the upper body. The landmarks are converted to body coordinates (origin between the "
        "shoulders, shoulder width equal to 1), which removes the effect of the camera's distance and image "
        "proportions, and each hand is also described relative to its own wrist so that hand shape is visible to "
        "the model. A sign starts when a hand rises above a rest line below the shoulders and ends when it is "
        "lowered; several signs made without lowering the hands are split where the hands slow down, keeping the "
        "cuts whose pieces the recognizer finds most convincing. The classifier is a one-dimensional convolutional "
        f"network combined with a transformer, following the Kaggle first-place solution {{ref:7}}, with "
        f"{r['params_M']:.2f} million parameters. Its scores are calibrated by temperature scaling {{cite:guo}} on "
        "signers held out for validation, and a threshold on the calibrated confidence decides whether a sign is "
        "accepted or its five most likely labels are shown. The component went through five versions "
        "(Section 7.5); each change was judged on validation signers only, and the test signers were used once "
        "per version.")
    b.body(
        "**Component Y: language and speech (Goal 2).** The sentence builder turns the accepted signs into English "
        "with templates for frequent service-counter phrases and a small language model for other sign sequences. "
        "The touchscreen shows the sentence together with the candidate signs, and only a confirmed sentence "
        "continues. Because the 1.2 s latency budget starts at the confirmation, this path is kept short: "
        "translation with NLLB-200 {ref:11} is skipped for English, template phrases can be translated and "
        "synthesized in advance, and Piper {ref:12} starts playing the first part of a sentence while the rest is "
        "still being produced. The reply channel uses voice-activity detection and Whisper {ref:13}, which "
        "transcribes a Turkish or Spanish reply and translates it into English in one step. The microphone is "
        "muted while the device speaks. First timings of these models are given in Section 7.5.")
    b.body(
        "**Component Z: embedded platform (Goal 3).** The platform is the Jetson Orin Nano developer kit with the "
        "peripherals of Fig. 6. The main design problem is to make all models share one 8 GB memory and one "
        "graphics processor. The models are exported to the Open Neural Network Exchange (ONNX) format and run "
        "with ONNX Runtime {cite:onnxruntime} or TensorRT at reduced precision (16-bit floating point or 8-bit "
        "integers), Whisper is loaded only while the hearing person speaks if memory runs short, and the power mode "
        "(7–25 W {cite:nvidia_specs}) is chosen from the measured frame rate, latency and temperature. The "
        "recognizer has already been exported to ONNX and checked against the original model (Section 7.5). The "
        "enclosure holds the camera and screen toward the Deaf user, turns the speaker and microphone toward the "
        "hearing person, and leaves room for air to flow around the module's heatsink and fan.")

    # ---------------------------------------------------------------- 7.4
    b.heading(2, "7.4 Hardware Block Diagram")
    b.body(
        "Fig. 6 is the hardware block diagram of the device. The blocks are its principal parts, grouped by the "
        "side of the counter they face, and the lines show how they are connected: solid lines carry data in the "
        "direction of the arrow, and dashed lines carry power. All processing happens on the Jetson Orin Nano "
        "module inside the base; every other block is an input, an output, storage or the power supply. Table 12 "
        "gives the proposed part, connection, function and estimated power of each block.")
    b.figure(os.path.join(FIG, "fig_hw_block_diagram.png"), 6.0,
             "Hardware block diagram. Light-emitting diode (LED), solid-state drive (SSD); interfaces as documented "
             "for the developer kit {cite:nvidia_guide}. The Wi-Fi card is removed and Ethernet is unplugged, so the "
             "device has no network connection in use.")
    b.body(
        "The connections follow from the developer kit's carrier board {cite:nvidia_guide}. Its only video output "
        "is DisplayPort, so the touchscreen is driven through a DisplayPort-to-HDMI adapter, while its touch panel "
        "connects over the Universal Serial Bus (USB). The camera, microphone and speaker also use the four USB 3.2 "
        "ports, which supply 5 V power as well as data; each pair of ports is limited to 3 A {cite:nvidia_guide}, "
        "enough for the display and the speaker. A push button and an LED use general-purpose input/output (GPIO) "
        "pins on the 40-pin header. The button is the backup for marking the end of a sign, and the LED is lit "
        "whenever the camera or microphone is on, so that everyone at the counter can see when the device is "
        "recording. The operating system and the models are stored on a 256 GB NVMe SSD in the M.2 Key-M slot, "
        "because the kit includes no storage of its own {cite:nvidia_guide}.")
    b.body(
        "Two blocks are deliberately not used. The kit ships with a Wi-Fi card in its M.2 Key-E slot "
        "{cite:nvidia_guide}; we remove it, and the Ethernet port stays unplugged except during development. This "
        "turns the offline requirement into a physical property of the device that can be checked by inspection "
        "(Section 7.7). Power comes from the 19 V adapter supplied with the kit {cite:nvidia_guide} or, optionally, "
        "from a battery pack with the same output, selected by a switch. With an estimated total of about 20 W in "
        "normal use and 40 W at peak (Table 12), a 100 Wh battery would last about 4.5 h of normal use, assuming "
        "90% conversion efficiency. These estimates will be replaced by the measurements of Goal 3.")
    b.table("Hardware blocks, connections and estimated power (power figures are estimates to be measured).",
            [1450, 2250, 1750, 2510, 1400],
            ["Block", "Proposed part", "Connection", "Function", "Power (W)"],
            [["Processing", "Jetson Orin Nano 8 GB developer kit", "—",
              "Runs all models: 6-core Arm CPU, 1024-core GPU, 8 GB shared memory {cite:nvidia_specs}",
              "7–25 (power mode)"],
             ["Camera", "USB RGB camera, 1080p, 30 frames/s", "USB 3.2", "Video of the signer for landmark "
              "extraction", "≈ 1–1.5"],
             ["Touchscreen", "7-inch HDMI touchscreen", "DisplayPort → HDMI; USB for touch",
              "Shows the sentence, candidates and reply; language selection", "≈ 3–4"],
             ["Button and LED", "Push button; LED with resistor", "GPIO, 40-pin header",
              "Backup sign boundary; recording indicator", "< 0.1"],
             ["Microphone", "USB directional microphone", "USB 3.2", "Records the hearing person's reply", "< 0.5"],
             ["Speaker", "USB-powered speaker", "USB 3.2", "Plays the spoken sentence", "≈ 1–5"],
             ["Storage", "256 GB NVMe SSD", "M.2 Key-M, PCIe 3.0", "Operating system, models and voices",
              "≈ 1–3.5"],
             ["Power", "19 V adapter (in the kit); optional 19 V battery pack, ≈ 100 Wh", "DC jack via a switch",
              "Supplies the kit, which powers the USB devices", "—"],
             ["Not used", "Wi-Fi card; Ethernet", "M.2 Key-E; RJ45", "Removed or unplugged for offline use", "0"],
             ["Total", "", "", "Normal use / peak", "≈ 20 / 40"]],
            bold_first_col=True, total_row=True)

    # ---------------------------------------------------------------- 7.5
    b.heading(2, "7.5 Preliminary Data and Simulations")
    b.body(
        "Most preliminary data come from Goal 1, which already has a working laptop prototype (P1 in Section 7.6). "
        "We also ran measurements and simulations that do not need the final hardware: the computing cost of the "
        "recognizer, first timings of the language and speech models on a cloud processor, a simulation of how "
        "often a user will have to correct the device, and a memory budget. All accuracy numbers below are for "
        "people who were never seen in training.")
    b.body(
        "**Recognition accuracy.** Fig. 7(a) shows the five versions of the recognizer. Versions v1–v4 were "
        "trained on the Kaggle data (250 signs, 21 signers) and tested on three signers not used in training. Each "
        "version added one group of changes: more augmentation (v2), the Kaggle first-place training recipe (v3), "
        f"and body coordinates with hand-shape features (v4). v4 reached {f1(r['v4_top1'])}% top-1 and "
        f"{f1(r['v4_top5'])}% top-5 accuracy, short of both targets. v5 is v4 fine-tuned on ASL Citizen for a "
        "134-word service-counter vocabulary and tested on its 11 Deaf test signers. It reaches "
        f"{f1(r['v5_top1'])}% top-1 and {f1(r['v5_top5'])}% top-5, above both targets of Table 3. The same model "
        f"trained from random weights reached {f1(r['scratch_top1'])}% and {f1(r['scratch_top5'])}%, so "
        "pretraining on the Kaggle data adds about one point here, but it shortens training and helps the weakest "
        "signers. Fig. 7(b) shows that the spread between signers shrank from 46–79% (v4 on Kaggle) to 79–89% "
        "(v5), and that the weakest v5 signer is above the 75% target. The most frequent remaining confusions are "
        "signs that look nearly the same or mean the same thing, such as HURT and PAIN, TOILET and BATHROOM, or "
        "COST and PRICE; these pairs will be merged into one spoken word, while pairs such as GOOD and THANK-YOU "
        "are left to the confirmation screen.")
    b.figure(os.path.join(FIG, "fig_prelim_results.png"), 6.3,
             "Preliminary recognition results on signers never seen in training. (a) Top-1 and top-5 accuracy of "
             "the five recognizer versions; v1–v4 were trained and tested on the Kaggle data (250 signs), and v5 "
             "was fine-tuned and tested on ASL Citizen (134 words). (b) Top-1 accuracy for each test signer; the "
             "horizontal bars are the means.")
    cal = B["calib"]
    b.body(
        "**Calibration and abstention.** Fig. 8 shows how well the confidence of v5 matches its accuracy. The raw "
        f"confidences are too low: temperature scaling with T = {cal['T']} lowers the expected calibration error "
        f"from {f1(cal['ece_raw'])}% to {f1(cal['ece_scaled'])}%. With a threshold of {cal['threshold']}, chosen on "
        f"the validation signers for a 5% error, the device accepts {f1(cal['coverage'])}% of the test signs and "
        f"{f1(cal['error'])}% of the accepted signs are wrong; the other signs go to the candidate list, which "
        f"contains the right sign {f1(r['v5_top5'])}% of the time. For the weakest signer, however, the same "
        f"threshold gives {f1(cal['worst_error'])}% error. A threshold fitted on a few signers does not hold for "
        "everyone, so it will be re-validated with our own camera and many signers (Section 7.7).")
    b.figure(os.path.join(FIG, "fig_calib_v5.png"), 6.0,
             "Calibration of the v5 recognizer on the 11 ASL Citizen test signers (134 words). (a) Reliability "
             "diagram: accuracy against confidence before (raw) and after temperature scaling; the diagonal is "
             "perfect calibration. (b) Risk–coverage curve: error among the signs the device answers against the "
             f"share of signs it answers. The chosen operating point answers {f1(cal['coverage'])}% of signs at "
             f"{f1(cal['error'])}% error.")
    b.body(
        "**Live tests and segmentation.** P1 runs at about 20 frames per second on a team member's laptop. Splitting "
        "several signs made without lowering the hands was correct in all three recordings tested (HELLO, WATER, "
        "PLEASE), but with a single signer this is not yet a reportable result. The first live test of v5 was worse "
        "than v4, although v5 is far better on ASL Citizen's recorded test signers. This points to a difference "
        "between the live pipeline and the training videos rather than to the model, and it is being "
        "investigated. It is also why recordings made with our own camera are part of the accuracy requirement "
        "(Table 16).")
    b.body(
        f"**Computing cost of the recognizer.** The v5 recognizer has {r['params_M']:.2f} million parameters "
        f"({f1(r['fp32_MB'])} MB at 32-bit precision) and needs about {r['macs_M']:.0f} million multiply-accumulate "
        "operations per sign. Exported to ONNX, it runs in "
        f"{f1(r['ort_ms_1t'])} ms per sign on one core of a cloud processor with ONNX Runtime {{cite:onnxruntime}}, "
        f"and its output matches the original model to within {r['max_abs_diff_str']}. The recognizer is therefore "
        "a negligible part of the time and memory budget; the landmark extractor and the language models dominate "
        "both.")
    tr_fp, tr_i8, es_fp, es_i8 = n["tr_fp32_s"], n["tr_int8_s"], n["es_fp32_s"], n["es_int8_s"]
    b.body(
        "**Timing of the language and speech models.** Table 13 gives first measurements of the Goal 2 models on a "
        f"{B['cpu_threads']}-core cloud processor ({cpu_label(B['cpu'])}), with ten typical service-counter sentences such as "
        "\"How much is this?\" and \"I want to pay with a credit card.\" These are not Jetson measurements, but they "
        "show where the 1.2 s budget goes. Converting NLLB-200 to 8-bit integers with CTranslate2 {cite:ctranslate2} "
        f"cut the median translation time from {tr_fp:.2f} s to {tr_i8:.2f} s for Turkish and from {es_fp:.2f} s to "
        f"{es_i8:.2f} s for Spanish, and shrank the model from {n['fp32_GB']:.1f} GB to {n['int8_GB']:.2f} GB. Piper "
        f"produced the first audio of a sentence within {p['first_max_s']:.2f} s in all three languages and "
        f"synthesized speech about {1 / p['en']['rtf']:.0f} times faster than real time. From confirmation to the "
        f"first audio, the median was therefore {sim['lat_tr_s']:.2f} s for Turkish and {sim['lat_es_s']:.2f} s for "
        "Spanish (Fig. 9(b)), within the 1.2 s budget before any use of the graphics processor. A first reading of "
        "the 20 translations found most of them correct, but three began with a stray dash and the Turkish word for "
        "\"appointment\" was misspelled. This supports the mitigation in Table 8: translations of template phrases "
        "will be checked by native speakers and stored, and the model will be used only for other sentences.")
    b.body(
        "For the reply channel we played the synthesized sentences back into Whisper. The small model transcribed "
        f"them with a word error rate of {w['small_wer_en']:.0f}% in English, {w['small_wer_es']:.1f}% in Spanish "
        f"and {w['small_wer_tr']:.1f}% in Turkish, and took about {w['small_en_s_tr']:.1f} s to turn a Turkish reply "
        "of about 1.5 s into English text; the base model was more than twice as fast but reached "
        f"{w['base_wer_tr']:.1f}% error in Turkish. Synthetic speech in a quiet room is an optimistic test, so the "
        "Turkish reply channel is the weakest link found so far. It will be tested with recorded voices and "
        "background noise, and a larger Whisper model on the graphics processor will be tried if needed.")
    b.table(f"First timing and size measurements on a {B['cpu_threads']}-core cloud processor (median over ten "
            "service-counter sentences; not yet on the Jetson).",
            [2500, 2860, 1900, 2100],
            ["Model", "Task", "Size", "Time (median)"],
            B["table13"], bold_first_col=False)
    b.body(
        "**Simulation of user corrections.** Fig. 9(a) uses the measured acceptance and error rates of each v5 test "
        "signer to estimate how often a sentence of *n* signs needs a correction, that is, a sign picked from the "
        "candidate list because it was not accepted, or a wrongly accepted sign replaced on the confirmation screen. "
        "Signs are assumed to be independent. For a three-sign sentence, the average signer would need no "
        f"correction {sim['p3_none_avg']:.0f}% of the time and {sim['e3_avg']:.1f} corrections on average; the "
        f"weakest signer would need {sim['e3_worst']:.1f}. A Monte Carlo run of 100,000 sentences per signer gave the "
        f"same mean ({sim['e3_mc_avg']:.2f}). Corrections are therefore a normal part of every "
        "conversation, and the confirmation screen must make them a single tap on a candidate rather than a "
        "repeated sign.")
    b.body(
        "**Memory budget.** Fig. 9(c) adds up the memory the device needs. The sizes of the recognizer, NLLB-200, the "
        "Piper voices and Whisper are measured from the model files; the operating system and desktop (2.0 GB), "
        "the graphics runtime (0.5 GB), the landmark extractor (0.1 GB) and the sentence builder's language model, "
        "which is not yet chosen (0.5 GB), are assumed allowances. The total, about "
        f"{f1(sim['mem_total_gb'])} GB, leaves about {f1(8 - sim['mem_total_gb'])} GB of the 8 GB for working memory "
        "and buffers, so all models should be able to stay loaded at the same time. This estimate will be checked "
        "on the Jetson (requirement R11 in Table 16).")
    b.figure(os.path.join(FIG, "fig_simulations.png"), 5.6,
             "Simulations and estimates based on measured data. (a) Expected corrections per sentence from the "
             "measured acceptance and error rates of the v5 test signers (mean signer and weakest signer). (b) "
             "Median time from confirmation to the first audio on a cloud processor, against the 1.2 s budget. (c) "
             "Memory budget of all models on the 8 GB Jetson Orin Nano; hatched parts are assumed allowances.")

    # ---------------------------------------------------------------- 7.6
    b.heading(2, "7.6 Prototyping Plan")
    b.body(
        "A prototype is an early version of a product, built to test a concept or to learn from it before the "
        "final version is made {cite:wiki_proto}. Prototypes are grouped into categories by what they represent: "
        "function, appearance, or the user's experience. We plan five prototypes, P1–P5, each in a different "
        "category and each built to answer specific questions as cheaply as possible (Table 14). Their order "
        "follows the risks of Table 8, so the riskiest questions are answered first. This is the idea of rapid "
        "prototyping: building part of a design quickly to test its riskiest aspects {cite:wiki_proto}.")
    b.body(
        "**P1: proof-of-principle prototype (built).** A proof-of-principle prototype verifies key functional "
        "aspects of a design without the rest of its functions {cite:wiki_proto}. P1 is the laptop demonstrator of "
        "Goal 1: a webcam, landmark extraction, sign segmentation and the recognizer, which shows the three most "
        "likely signs on screen. It has shown that one ordinary camera and a small landmark model can recognize "
        "signs live, and its results are in Section 7.5.")
    b.body(
        "**P2: user-experience prototype (paper and Wizard of Oz).** A user-experience prototype represents enough "
        "of the appearance and function of a product to be used for user research, and a paper prototype is a "
        "drawn version of a software interface used for early testing {cite:wiki_proto}. P2 consists of printed and "
        "then clickable mock-ups of the confirmation screen, tested with Deaf ASL users before the Goal 2 software "
        "is finished. In a Wizard of Oz prototype, a hidden person plays the part of the computer "
        "{cite:wiki_proto}; a team member will play the sentence builder and the voice, so that the full "
        "conversation can be tried before it is built. P2 answers whether users understand the screen and can "
        "correct a sentence quickly, which Fig. 9(a) shows they will need to do often.")
    b.body(
        "**P3: functional bench prototype.** An electronics prototype, often built on a breadboard, is electrically "
        "like the final design but not physically like it {cite:wiki_proto}. P3 is the Jetson developer kit with all "
        "USB peripherals on the bench, without an enclosure. It answers the main Goal 3 questions: whether "
        "MediaPipe runs on the Jetson at 15 frames per second or more, whether all models fit in 8 GB, whether the "
        "1.2 s latency is met, and how much power the device draws.")
    b.body(
        "**P4: visual (form-study) prototype.** A visual prototype represents the size and appearance of a "
        "product but not its function, and a form study emphasizes its geometry {cite:wiki_proto}. P4 is an "
        "enclosure without electronics, first in cardboard and then 3D-printed, placed on a real counter. It checks "
        "the screen angle, the camera's view of a signer, the direction of the speaker and microphone, and the "
        "space for vents and cables.")
    b.body(
        "**P5: working prototype (alpha, then beta).** A working prototype represents all or nearly all of the "
        "functions of the final product {cite:wiki_proto}. P5 combines the electronics of P3 with the enclosure of "
        "P4 and the complete software. Its software follows the usual stages {cite:wiki_proto}: an alpha version "
        "that runs end to end with few functions (English only, template sentences only), then a feature-complete "
        "beta version used for verification and for validation with Deaf ASL users.")
    b.table("Prototypes of the project and their categories.",
            [700, 1850, 2300, 2810, 1700],
            ["No.", "Category", "What is built", "Questions it answers", "Status (Oct. 2026)"],
            [["P1", "Proof of principle (technology demonstrator)",
              "Laptop and webcam: landmarks, segmentation, recognizer, top-3 display",
              "Can one RGB camera and a landmark model recognize signs live?", "Built; v5 at 86.6% top-1 on "
              "unseen signers"],
             ["P2", "User experience: paper and Wizard of Oz",
              "Printed, then clickable, confirmation screen; a team member plays the sentence builder and voice",
              "Do Deaf users understand the screen? How fast can they correct a sentence?",
              "Planned with the NJIT ASL community"],
             ["P3", "Functional bench (breadboard-style)", "Jetson developer kit with USB peripherals, no enclosure",
              "Frame rate, memory, latency and power on the real hardware", "After the purchase"],
             ["P4", "Visual and form study", "Cardboard, then 3D-printed, enclosure without electronics",
              "Screen angle, camera view, speaker and microphone direction, ventilation", "Planned"],
             ["P5", "Working prototype (alpha, then beta)", "P3 electronics in the P4 enclosure with all software",
              "Does the device meet Table 16, and do Deaf users find it useful?", "ECE 416"]],
            bold_first_col=True)
    b.body(
        "Each prototype ends with a short review against Tables 3 and 8. A target that is met closes the related "
        "risk; a missed target sends the work back to the design concept, which is the dashed line in Fig. 5.")

    # ---------------------------------------------------------------- 7.7
    b.heading(2, "7.7 Verification and Validation")
    b.body(
        "Verification asks whether the device is built right, that is, whether it meets its specifications; "
        "validation asks whether the right device is built, that is, whether it meets the needs of its users and "
        "other stakeholders {cite:wiki_vv}. Both are needed. A recognizer can meet its accuracy target on a dataset "
        "and still fail Deaf users at a real counter, as the first live test of v5 already showed. Our plan follows "
        "IEEE 1012 {ref:26} and the V-model of Fig. 10: each design level on the left is checked by a test level on "
        "the right, from unit tests of each component up to validation with Deaf ASL users, and the prototypes of "
        "Section 7.6 are the objects being tested at each level.")
    b.figure(os.path.join(FIG, "fig_vmodel.png"), 6.0,
             "V-model of verification and validation. Each design level (left) is verified or validated by the "
             "test level opposite it (right); the prototypes used at each level are named between them. P2 "
             "validates the user interface early, before P5 exists.")
    b.body(
        "**Verification methods.** Each requirement is verified by one or more of four standard methods "
        "{cite:nasa}: *test*, a measurement under controlled conditions; *analysis*, a calculation or simulation; "
        "*inspection*, an examination of the design, code or device; and *demonstration*, showing that a function "
        "works without measuring it. Table 16 is the verification matrix of the project. It gives each requirement "
        "of Tables 1, 3 and 5 and of Section 6.1 an identifier, its method, its pass criterion and the preliminary "
        "result, so it also serves as a status report.")
    b.body(
        "**Validation and qualification categories.** Table 15 lists the categories of validation and "
        "qualification {cite:wiki_vv} and how each applies to this project. Most of our results so far are "
        "*retrospective*: they check the recognizer against recorded datasets. Before any result is reported, we "
        "also fix the test protocol, the pass criteria and the threshold in advance and use the test data only "
        "once (*prospective* validation). *Re-validation* matters especially for a learned model: every retraining, "
        "vocabulary change, new threshold, software update or new camera can change the results, so a fixed set of "
        "regression tests is re-run after each change. For the hardware, the qualification stages check that the "
        "purchased parts meet their specification (design qualification), that the device is set up correctly "
        "(installation qualification), that every function works within its limits (operational qualification) and "
        "that the device performs in real use (performance qualification).")
    b.table("Validation and qualification categories and how the project applies them.",
            [1900, 2900, 3160, 1400],
            ["Category", "Meaning", "Application in this project", "When"],
            [["Validation", "", "", ""],
             ["Retrospective", "Checking against existing, recorded data",
              "Recognizer tested on Kaggle and ASL Citizen signers never seen in training (Figs. 7, 8)",
              "Done for v1–v5"],
             ["Prospective", "Done before release, with criteria fixed in advance",
              "Protocol, pass criteria and threshold fixed on validation signers before the test set is opened; "
              "test set used once", "Before every reported result"],
             ["Partial", "A limited test of the most important effects, e.g. a pilot",
              "Pilot checks such as fluent splitting on three recordings and first live tests; not reported as "
              "results", "Ongoing"],
             ["Concurrent", "Done during normal operation",
              "The device logs times, confidences and corrections (never video or audio) during sessions with "
              "Deaf signers", "P5 sessions"],
             ["Cross-validation", "Confirming a result with independent data or a second method",
              "Same model on Kaggle and ASL Citizen; pretrained against trained from scratch; our own camera "
              "recordings as a third data set", "Done; own data planned"],
             ["Re-validation", "Repeated after a change, repair or move",
              "Regression tests after retraining, a vocabulary, threshold or software change, or a new camera",
              "After every change"],
             ["Qualification", "", "", ""],
             ["Design (DQ)", "Purchased equipment meets the written specification",
              "Jetson, camera, display, microphone and speaker checked against Tables 6 and 12", "Before purchase"],
             ["Installation (IQ)", "The equipment is installed and configured correctly",
              "Operating system version, model checksums, Wi-Fi card removed, networking off", "P3 and P5"],
             ["Operational (OQ)", "Every function works within its operating limits",
              "Frame rate, latency, memory, temperature and power in each power mode", "P3 and P5"],
             ["Performance (PQ)", "The system performs under real conditions",
              "Mock service counter with Deaf signers, background noise and different lighting", "P5"]],
            group_rows=(0, 7), bold_first_col=False)
    b.body(
        "**Validation with Deaf ASL users.** The final validation uses P5 at a mock service counter. Deaf and fluent "
        "ASL signers from the NJIT community (target: at least five people, depending on recruitment and the IRB "
        "review of Section 6.2) will complete short scripted tasks with a hearing staff member, such as ordering an "
        "item, asking a price and asking for directions. We will measure whether each task is completed, the time "
        "per sentence, the number of corrections, the accuracy of the spoken sentences and, through a short "
        "questionnaire in ASL or English, whether the participants would use the device. Their feedback on the "
        "vocabulary and the screen goes back into the design concept, as in Fig. 5.")
    b.table("Verification matrix (T = test, A = analysis, I = inspection, D = demonstration).",
            [560, 2280, 900, 2720, 2900],
            ["ID", "Requirement", "Method", "Pass criterion", "Preliminary result (Oct. 2026)"],
            [["R1", "Top-1 accuracy on unseen signers", "T", "≥ 75% on the ASL Citizen test split and on "
              "recordings from our own test users", f"{f1(r['v5_top1'])}% on ASL Citizen (11 signers); own "
              "recordings pending"],
             ["R2", "Top-5 accuracy on unseen signers", "T", "≥ 90%, same data as R1",
              f"{f1(r['v5_top5'])}% on ASL Citizen; own recordings pending"],
             ["R3", "Accuracy reported for each signer", "I", "Every test signer listed", "79.3–88.9% (Fig. 7(b))"],
             ["R4", "Accuracy of accepted signs", "A", "Risk–coverage curve and operating point reported",
              f"{f1(cal['coverage'])}% accepted at {f1(cal['error'])}% error (Fig. 8)"],
             ["R5", "Vocabulary of 100–250 signs", "I", "Final list reviewed with Deaf ASL users",
              "Draft list of 134 words; review pending"],
             ["R6", "Nothing spoken without confirmation", "T, I",
              "No speech without a confirmation event in any test log; code review", "Planned (P2, P5)"],
             ["R7", "Latency from confirmation to speech", "T", "< 1.2 s for every test sentence",
              f"{sim['lat_tr_s']:.2f} s (TR) and {sim['lat_es_s']:.2f} s (ES) median on a CPU; Jetson pending"],
             ["R8", "Turkish and Spanish output quality", "T", "Rated acceptable by native speakers",
              "Sample translations produced; rating pending"],
             ["R9", "Reply shown as readable English", "T", "Word error rate reported per language",
              f"{w['small_wer_en']:.0f}% / {w['small_wer_tr']:.1f}% / {w['small_wer_es']:.1f}% (EN / TR / ES) on "
              "synthetic speech; real voices pending"],
             ["R10", "Frame rate on the Jetson", "T", "≥ 15 frames/s for landmarks and recognition",
              "About 20 frames/s on a laptop; Jetson pending"],
             ["R11", "All models within 8 GB", "A, T", "Estimate, then measured peak use below 8 GB",
              f"Estimate {f1(sim['mem_total_gb'])} GB (Fig. 9(c))"],
             ["R12", "Fully offline operation", "D, I", "Full conversation with Wi-Fi removed and networking off",
              "Planned (IQ)"],
             ["R13", "No video or audio stored or sent", "I, T",
              "No media files written and no network traffic during a session", "Planned (IQ, OQ)"],
             ["R14", "Power draw measured", "T", "Idle and active power in each power mode", "Planned (P3)"],
             ["R15", "Safe surface temperature", "T", "Enclosure within the touch-temperature limits of ANSI/UL "
              "62368-1 {ref:28} after 1 h at full load", "Planned (P5)"]],
            bold_first_col=True, center_cols=(2,))

    # ---------------------------------------------------------------- 7.8
    b.heading(2, "7.8 Prototype Assembly and Integration")
    b.body(
        "The final prototype is assembled step by step, so that a problem can always be traced to the last part "
        "added. After the purchase, the Jetson is first set up and checked (installation qualification), and "
        "landmark extraction is tested on it before anything else, because it is the highest Goal 3 risk in "
        "Table 8. The recognizer is added next, followed by the sentence builder and the touchscreen, then "
        "translation and speech, and finally the reply channel. After each addition the frame rate, memory use and "
        "latency are measured again; if a limit is exceeded, the last component is optimized before the next one is "
        "added. The electronics are then mounted in the enclosure designed with P4, with strain relief on the "
        "cables and vents around the module, as Section 6.1 requires. The complete device, P5, is verified against "
        "Table 16 and validated with Deaf ASL users, and the results will be reported in ECE 416.")
