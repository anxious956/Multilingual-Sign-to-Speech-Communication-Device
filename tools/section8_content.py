"""Text, tables and references of proposal Section 8, Relevant Technical Background (review of articles and
patents), used by
tools/build_proposal.py. Markup as in section7_content.py.

Bibliographic data checked on 26 Oct 2026: patents on Google Patents (title, number, abstract, dates, legal status;
abstracts are quoted verbatim), articles through Crossref and DataCite (DOI, ISBN, pages).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")

REFERENCES = {
    "sorenson": "Carlock, J., and Maxwell, C. Methods, systems, and machine-readable media for translating sign "
                "language content into word content and vice versa. *U.S. Patent* No. 12,131,586 B2, 2024.",
}

REC_W = [2150, 7210]  # bibliographic record tables: field | details


def update_earlier_sections(b):
    """Reference list: granted numbers for [19] and [20], verified pages and DOIs for [1] and [2]; Section 6.2:
    the freedom-to-operate point raised by the review of the Sign-Speak patent."""
    b.replace_text("1.\tTran", "(ASSETS ’23), New York, NY, USA, 2023.", "(ASSETS ’23), 1–7, New York, NY, USA, 2023.")
    b.replace_text("2.\tDesai", ", 36 (Datasets and Benchmarks Track), 2023. arXiv:2304.05934.",
                   ", 36 (Datasets and Benchmarks Track), 76893–76907, 2023. doi:10.52202/075280-3360.")
    b.replace_text("19.\tKelly", "U.S. Patent Application Publication", "U.S. Patent")
    b.replace_text("19.\tKelly", " No. 2022/0327961 A1, 2022.",
                   " No. 12,518,653 B2, 2026 (published as application No. 2022/0327961 A1, 2022).")
    b.replace_text("20.\tZhou", "U.S. Patent Application Publication", "U.S. Patent")
    b.replace_text("20.\tZhou", " No. 2016/0307469 A1, 2016.",
                   " No. 10,109,219 B2, 2018 (published as application No. 2016/0307469 A1, 2016).")
    b.replace_text("Legal and regulatory constraints.", "and safety certification (Section 6.1).",
                   "and safety certification (Section 6.1). It would also need a freedom-to-operate review of the "
                   "active patents in Section 8, in particular the granted Sign-Speak patent [19], whose main claim "
                   "covers pose-based sign recognition in general terms.")


def record(b, caption, rows):
    b.table(caption, REC_W, ["Item", "Details"], rows, bold_first_col=True)


def build(b):
    b.page_break()
    b.heading(1, "8. Relevant Technical Background")  # title of the ECE 414 assignment
    b.body(
        "This section reviews in detail the two research articles and the four patents that are closest to the "
        "proposed device, and explains for each how our proposal differs from it. Articles were found with Google "
        "Scholar (https://scholar.google.com) and patents with Google Patents (https://patents.google.com), using "
        "combinations of the terms sign language recognition, sign language translation, isolated sign, pose, "
        "camera, speech and confidence. Patents were selected for being close to our design and for representing "
        "the main approaches: a single camera with cloud processing (Sign-Speak), multiple cameras with a depth "
        "sensor (Avodah), translation inside a video relay service (Sorenson) and sensor gloves (Bosch). For every "
        "patent the abstract is quoted from Google Patents, and the legal status was checked there on 26 October "
        "2026. For the articles, the identifiers (DOI, ISBN and pages) were confirmed with the publishers' records. "
        "Section 8.4 compares all of them with the proposed device.")

    # ---------------------------------------------------------------- 8.1 articles
    b.heading(2, "8.1 Article 1: ASL Citizen Dataset")
    record(b, "Bibliographic record of Article 1.", [
        ["Title", "ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition"],
        ["Authors", "Aashaka Desai, Lauren Berger, Fyodor O. Minakov, Vanessa Milan, Chinmay Singh, Kriston "
                    "Pumphrey, Richard E. Ladner, Hal Daumé III, Alex X. Lu, Naomi Caselli, Danielle Bragg"],
        ["Published in", "Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and "
                         "Benchmarks Track, pp. 76893–76907, 2023 {ref:2}"],
        ["ISBN / ISSN", "ISBN 978-1-7138-9911-2 (proceedings volume 36); ISSN 1049-5258"],
        ["DOI", "10.52202/075280-3360 (proceedings); 10.48550/arXiv.2304.05934 (preprint)"],
        ["Links", "https://doi.org/10.52202/075280-3360; https://arxiv.org/abs/2304.05934; dataset: "
                  "https://www.microsoft.com/en-us/research/project/asl-citizen/"],
    ])
    b.body(
        "**Description.** The article introduces ASL Citizen, the first crowdsourced isolated sign language "
        "recognition dataset collected with the consent of its participants. It contains 83,399 videos of 2,731 "
        "distinct ASL signs, recorded by 52 Deaf and hard-of-hearing signers with their own webcams in everyday "
        "environments, about 30 videos per sign. Recruitment and consent materials were presented in ASL first, and "
        "participants were compensated. The authors frame isolated sign recognition as dictionary retrieval: a user "
        "demonstrates a sign to a webcam and the system returns a ranked list of matching signs. The 52 signers are "
        "divided into fixed training, validation and test sets so that test signers are never seen in training, "
        "and results are reported as recall at 1, 5 and 10 (recall at 1 is ordinary accuracy), discounted cumulative "
        "gain and mean reciprocal rank.")
    b.body(
        "Two baselines are trained. An appearance model, I3D, a three-dimensional convolutional network applied to "
        "the video frames, reaches 63.10% top-1, 86.09% top-5 and 90.86% top-10 accuracy on the unseen test "
        "signers. A pose model, ST-GCN, which works on MediaPipe Holistic keypoints, reaches 59.52% top-1. The "
        "authors also show that the source of the training data matters more than the model: on the signs shared "
        "with the older WLASL dataset, a model trained on ASL Citizen reaches 74.16% top-1 against 8.49% for the same "
        "model trained on WLASL. The data may be used for non-commercial research only.")
    b.body(
        "**Difference from our proposal and how we use it.** ASL Citizen is a dataset and benchmark, not a device, "
        "so it is a resource for our project rather than a competing design. It is the main training and test data "
        "of our recognizer (Section 7.5): our v5 model, fine-tuned on a 134-word service-counter subset of it, "
        "reaches 86.6% top-1 and 98.2% top-5 on the same unseen test signers. The two results are not directly "
        "comparable, because the article ranks all 2,731 signs while our device chooses among 134 words. Beyond the "
        "dataset, our proposal adds what a deployed device needs and the article does not address: a small "
        "pose-based model that runs offline on embedded hardware (1.78 million parameters, about 4 ms per sign on one "
        "processor core), calibrated confidence with abstention and risk–coverage evaluation, accuracy reported for "
        "each signer, and the language and speech path from recognized signs to a spoken sentence. Its use case, "
        "dictionary look-up, is also different from ours, a conversation at a service counter.")

    b.heading(2, "8.2 Article 2: U.S. Deaf Community Perspectives on Automatic Sign Language Translation")
    record(b, "Bibliographic record of Article 2.", [
        ["Title", "U.S. Deaf Community Perspectives on Automatic Sign Language Translation"],
        ["Authors", "Nina Tran, Richard E. Ladner, Danielle Bragg"],
        ["Published in", "Proceedings of the 25th International ACM SIGACCESS Conference on Computers and "
                         "Accessibility (ASSETS ’23), New York, NY, USA, October 22–25, 2023, pp. 1–7 {ref:1}"],
        ["ISBN", "979-8-4007-0220-4 (ASSETS ’23 proceedings, ACM)"],
        ["DOI", "10.1145/3597638.3614507"],
        ["Links", "https://dl.acm.org/doi/10.1145/3597638.3614507; https://www.microsoft.com/en-us/research/"
                  "publication/u-s-deaf-community-perspectives-on-automatic-sign-language-translation/"],
    ])
    b.body(
        "**Description.** The article reports a survey of 32 Deaf and hard-of-hearing ASL users in the United "
        "States about automatic sign language translation. Participants completed an ASL check, and ASL was the "
        "primary language of 81% of them. For 14 everyday settings they rated how often they had wanted an "
        "interpreter but could not get one, and how willing they would be to use automatic translation. The unmet "
        "need was largest in medical, professional and educational settings, while willingness was highest for "
        "going into a business, followed by self-service kiosks and informal education; attitudes toward medical "
        "use were sharply divided. Participants expected the highest accuracy in medical settings and the lowest "
        "for kiosk-type exchanges. On form factors, 91% were comfortable with a phone app, 69% with a stand-alone "
        "system such as a computer on a counter, 38% with a wearable and 6% with an implant.")
    b.body(
        "The survey also measured concerns. More than 80% of participants were concerned about hearing people "
        "profiting from ASL, systems missing ASL grammatical features, automatic translation weakening the legal "
        "right to other accommodations, poorer performance for some users than for others, and limited Deaf "
        "involvement in leadership. The most valued benefit was access in more places and at the last minute. The "
        "authors conclude that developers should target specific settings chosen with the community and should "
        "involve Deaf people throughout.")
    b.body(
        "**Difference from our proposal and how we use it.** This article is a user study, not a technical system, "
        "and it is the main evidence behind several of our design decisions (Sections 3.3 and 5). Following its "
        "results, our device targets the business and service-counter setting, where willingness is highest and "
        "accuracy expectations are lower; it is a stand-alone counter device, the form factor 69% of participants "
        "accepted, rather than a wearable; it reports accuracy for each signer; and it is presented as an aid when "
        "no interpreter is available rather than as a replacement. The article goes further than our device in one "
        "respect: 91% of participants were concerned about systems that miss ASL grammar, and our device "
        "recognizes individual signs without facial grammar. We state this limitation openly instead of claiming "
        "full translation.")

    # ---------------------------------------------------------------- 8.3 patents
    b.heading(2, "8.3 Patents")
    b.body(
        "Four patents are reviewed in detail below, from the closest to our design to the most distant. Fig. 11 "
        "reproduces one drawing from each of the three closest. An expired Hitachi patent {ref:21} is "
        "summarized at the end for completeness.")
    b.figure(os.path.join(FIG, "fig_patent_drawings.png"), 6.2,
             "Drawings reproduced from the reviewed patents. (a) Sign-Speak, FIG. 7: after the user signs, the three "
             "most likely sentences are shown for confirmation {ref:19}. (b) Avodah, FIG. 4: one device between two "
             "people, with sensors, two displays and a speaker {ref:18}. (c) Sorenson, FIG. 3: a video relay "
             "service in which artificial intelligence (AI) servers turn ASL video into voice and voice into "
             "text, with a training station {cite:sorenson}.")

    # Patent 1: Sign-Speak
    b.item_heading("Patent 1: Realtime AI Sign Language Recognition (Sign-Speak)")
    record(b, "Record of Patent 1.", [
        ["Title", "Realtime AI sign language recognition"],
        ["Number", "U.S. Patent No. 12,518,653 B2, granted January 6, 2026 {ref:19}; published as application "
                   "US 2022/0327961 A1 on October 13, 2022"],
        ["Inventor / assignee", "Nikolas Anthony Kelly / Sign-Speak Inc."],
        ["Dates", "Priority May 11, 2020; filed May 11, 2021"],
        ["Legal status", "Active; Google Patents gives January 23, 2043 as the expected expiration"],
        ["Link", "https://patents.google.com/patent/US12518653B2/en"],
    ])
    b.body(
        "**Abstract.** \"A real time sign language recognition method that allows Deaf and Hard of Hearing "
        "individuals to sign into any apparatus with a camera to extract target information (such as a translation "
        "in a target language) is proposed.\"")
    b.body(
        "**Description.** The patent addresses conversations between Deaf signers and non-signers in impromptu "
        "settings where an interpreter cannot be provided immediately. It criticizes earlier systems that need "
        "gloves, 3D cameras or camera arrays, per-user calibration, or heavy computation on the device, and instead "
        "requires \"only a device with internet connection and a single lens camera\". Its general architecture has "
        "four stages, a capture device, a feature extractor, a processing module and an output device, and at least "
        "the feature extractor or the processing module must run, at least in part, on a cloud server.")
    b.body(
        "In the translation embodiment, a convolutional pose machine estimates the body pose, a region-based "
        "convolutional network finds the hands, and a second pose network gives the keypoints of the dominant and "
        "non-dominant hands. These are merged into one flattened feature vector per frame, normalized so that the "
        "head is at the origin and the shoulders are one unit apart, smoothed and appended to a feature queue. A "
        "one-dimensional convolutional network splits the queue into sign, transition and non-signing regions. A "
        "comparator first looks for the whole signed passage in a base of pre-recorded sentences using k-nearest "
        "neighbours with dynamic time warping; if none matches, each sign is matched in the same way against a base "
        "of single signs, and fingerspelled regions go to a separate sequence model. The resulting glosses are "
        "translated into English by a sequence-to-sequence grammar model and returned to the device as text. Other "
        "embodiments highlight the person who is signing in a video conference, answer signed questions, and show "
        "an \"ASL menu\" of the three most likely sentences from the sentence base for the user to confirm (Fig. "
        "11(a)). The granted claim 1 is broader than the published application: it claims capturing images, a "
        "first pose network for the body and a second for the hands, a flattened feature vector of fixed size, a "
        "feature queue, and its conversion into a target-language output, without requiring a server.")
    b.body(
        "**Difference from our proposal.** Sign-Speak is the closest prior art. Both systems use one ordinary "
        "camera, body and hand pose features normalized to the body, sign splitting, a fingerspelling path and a "
        "list of candidates that the user confirms. Our proposal differs in four ways. First, Sign-Speak needs a "
        "network connection and sends the user's data to a server, while our device runs fully offline on an "
        "embedded GPU and no video leaves it. Second, Sign-Speak matches signs to stored examples by nearest "
        "neighbour, while we use a trained neural classifier whose output is calibrated as a probability. Third, "
        "Sign-Speak's candidate menu is a separate mode that always offers three whole sentences from a fixed "
        "bank, while our device decides for each sign, from its calibrated confidence, whether to accept it or to "
        "show five candidates, always asks for confirmation of the complete sentence before speaking, and measures "
        "this trade-off with risk–coverage curves (Fig. 8). Fourth, Sign-Speak returns English text, while our "
        "device speaks English, Turkish or Spanish and shows the hearing person's spoken reply as text. One caution "
        "follows from the review: because the granted claim 1 describes pose-based recognition with body and hand "
        "pose networks in general terms, a pipeline like ours, which uses the MediaPipe body and hand models and a "
        "sequence of flattened per-frame features, may fall within its scope even though the differences above are "
        "real. The differences establish the originality of the proposed system but do not by themselves avoid the "
        "claim, so a commercial version would need a freedom-to-operate review (Section 6.2).")

    # Patent 2: Avodah
    b.item_heading("Patent 2: Automated Sign Language Translation Using Multiple Input and Output Modalities (Avodah)")
    record(b, "Record of Patent 2.", [
        ["Title", "Automated sign language translation and communication using multiple input and output modalities"],
        ["Number", "U.S. Patent No. 10,489,639 B2, granted November 26, 2019 {ref:18}; continuations include "
                   "US 10,956,725 B2 and US 12,183,123 B2"],
        ["Inventors / assignee", "Michael Menefee, Dallas Nash, Trevor Chandler / Avodah, Inc. (originally Avodah "
                                 "Labs, Inc.)"],
        ["Dates", "Priority February 12, 2018; filed January 25, 2019"],
        ["Legal status", "Active; the four-year maintenance fee was paid in 2023"],
        ["Link", "https://patents.google.com/patent/US10489639B2/en"],
    ])
    b.body(
        "**Abstract.** \"Methods, apparatus and systems for recognizing sign language movements using multiple input "
        "and output modalities. One example method includes capturing a movement associated with the sign language "
        "using a set of visual sensing devices, the set of visual sensing devices comprising multiple apertures "
        "oriented with respect to the subject to receive optical signals corresponding to the movement from "
        "multiple angles, generating digital information corresponding to the movement based on the optical signals "
        "from the multiple angles, collecting depth information corresponding to the movement in one or more planes "
        "perpendicular to an image plane captured by the set of visual sensing devices, producing a reduced set of "
        "digital information by removing at least some of the digital information based on the depth information, "
        "generating a composite digital representation by aligning at least a portion of the reduced set of digital "
        "information, and recognizing the movement based on the composite digital representation.\"")
    b.body(
        "**Description.** The patent describes a device that lets a signer and a non-signer communicate in both "
        "directions. The device has a set of sensor inputs and outputs for each party; it identifies the language "
        "each person uses and outputs the translation in the form the other person prefers: text, audible speech, "
        "or a visual language such as glyphs, animations or a signing avatar, possibly in a different sign "
        "language. It can be used face to face (Fig. 11(b): sensors facing each person, a processor, a primary and a "
        "secondary display and a speaker) or by two parties in different places connected over a network. "
        "Translation can be performed \"locally, remotely or both\", with cloud resources used, for example, when "
        "many output sign languages must be supported.")
    b.body(
        "The claims focus on how the signing is captured. Several visual sensors with multiple apertures view the "
        "signer from different angles, and a depth sensor, either structured light or time of flight, measures depth "
        "perpendicular to the image plane. The depth is used to discard part of the image data, for example the "
        "background, and the remaining views are aligned into one composite three-dimensional representation, from "
        "which a neural network recognizes the movement. Dependent claims add the capture of eyebrow movements, gaze "
        "direction and blinks, which carry grammatical information in sign languages. The patent has 30 claims, and "
        "Avodah holds a family of related patents on gesture recognition, training and user interfaces.")
    b.body(
        "**Difference from our proposal.** Avodah's face-to-face embodiment is the closest to our device in layout: "
        "one unit between two people, with inputs and screens facing each side and a speaker (compare Fig. 11(b) "
        "with Fig. 2). The technical approach, however, is different. Every independent claim requires multiple "
        "camera apertures and a depth sensor, while our device uses one ordinary RGB camera and no depth sensor, "
        "which makes it cheaper and simpler and keeps it outside those claims. Avodah also captures facial "
        "grammar, which we do not, and outputs signing avatars and several sign languages, while our device "
        "recognizes ASL only and speaks three spoken languages. Offline operation is optional in Avodah's design and "
        "mandatory in ours, and Avodah does not describe abstaining or confirming before output, which are central "
        "to our design.")

    # Patent 3: Sorenson
    b.item_heading("Patent 3: Translating Sign Language Content into Word Content and Vice Versa (Sorenson)")
    record(b, "Record of Patent 3.", [
        ["Title", "Methods, systems, and machine-readable media for translating sign language content into word "
                  "content and vice versa"],
        ["Number", "U.S. Patent No. 12,131,586 B2, granted October 29, 2024 {cite:sorenson}"],
        ["Inventors / assignee", "John Carlock, Conrad Maxwell / Sorenson IP Holdings, LLC"],
        ["Dates", "Priority and filing April 9, 2021"],
        ["Legal status", "Active; Google Patents gives February 12, 2042 as the expected expiration"],
        ["Link", "https://patents.google.com/patent/US12131586B2/en"],
    ])
    b.body(
        "**Abstract.** \"Communication systems, methods, and non-transitory machine-readable storage media are "
        "disclosed herein. A communication system may include a communication device configured to receive a video "
        "stream including sign language content and any content indicators associated with the video stream during "
        "a real-time communication session within a single communication device or between a plurality of "
        "communication devices. The communication system may also include a translation engine configured to "
        "automatically translate the sign language content into word content during the real-time communication "
        "session without assistance of a human sign language interpreter. Further, the communication system may be "
        "configured to output the word content translation to a communication device during the real-time "
        "communication session.\"")
    b.body(
        "**Description.** Sorenson provides video relay services, in which a Deaf person signs over video to an "
        "interpreter who voices the call to a hearing person. This patent describes replacing or assisting that "
        "interpreter with a translation engine. The engine divides the incoming video into sign language segments "
        "and compares each with stored video files, each linked to word content, and combines the matched words "
        "into a translation. It uses context in two ways: words already translated earlier in the session, and "
        "\"content indicators\", the facial expressions, body language and other visual cues that accompany the "
        "hand and arm movements and can change the meaning, express emotion or mark punctuation. The engine also "
        "works in reverse, producing sign language video from words.")
    b.body(
        "Three configurations are described: a single device, such as a cell phone, that acts as a personal "
        "translator with an on-board translation engine; several devices connected over a network; and a video "
        "relay service in which servers translate ASL video into voice for the hearing party and the hearing "
        "party's voice into text for the Deaf user (Fig. 11(c)). In the relay configuration a training station, "
        "operated by a human who knows sign language or by an AI training engine, monitors sessions and updates "
        "the engine's databases to improve accuracy. The independent claim covers a communication device and a "
        "translation engine that translates each segment using the words already determined from earlier segments "
        "and the content indicators.")
    b.body(
        "**Difference from our proposal.** This patent shows that a major provider of interpreting by video "
        "relay is itself pursuing automatic translation, which confirms the commercial interest in the field. Its "
        "setting, however, is a remote call through a relay service, while ours is a face-to-face conversation at a "
        "counter, where a relay service is not involved. It aims at continuous translation that uses context and "
        "facial cues, which is more ambitious than our recognition of single signs from a fixed vocabulary, and it "
        "relies on servers and a monitoring trainer, while our device works alone and offline. The patent does not "
        "describe a confidence threshold or a confirmation step: the translation is passed to the other party "
        "during the session. In our device the Deaf user sees and confirms every sentence before it is spoken, and "
        "uncertain signs are offered as a list of candidates.")

    # Patent 4: Bosch
    b.item_heading("Patent 4: System and Method for Automated Sign Language Recognition (Bosch)")
    record(b, "Record of Patent 4.", [
        ["Title", "System and method for automated sign language recognition"],
        ["Number", "U.S. Patent No. 10,109,219 B2, granted October 23, 2018 {ref:20}; published as application "
                   "US 2016/0307469 A1; related EP 3284019 B1, CN 107690651 B, WO 2016/168591 A1"],
        ["Inventors / assignee", "Zhengyu Zhou, Tobias Menne, Kehuang Li, Kui Xu, Zhe Feng, Chin-Hui Lee / Robert "
                                 "Bosch GmbH"],
        ["Dates", "Priority April 16, 2015; filed April 15, 2016"],
        ["Legal status", "Active; the eight-year maintenance fee was paid in April 2026"],
        ["Link", "https://patents.google.com/patent/US10109219B2/en"],
    ])
    b.body(
        "**Abstract.** \"A method for sign language recognition includes receiving, with an input device, an input "
        "based on a plurality of hand movements and postures of a user that correspond to a sequence of signs, "
        "extracting a plurality of features from the input corresponding to the plurality of hand movements and "
        "postures, identifying, a start of the sequence of signs in the input based on a first set of features in "
        "the plurality of features and a first Hidden Markov Model (HMM) stored in the memory and identifying a first "
        "sign in the input based on a second set of features in the plurality of features and a second HMM stored "
        "in the memory. The method also includes generating an output corresponding to the first sign from the "
        "input.\"")
    b.body(
        "**Description.** The patent applies the methods of speech recognition to continuous sign language. Its "
        "input comes from \"sensor signals from wearable device and/or image sequences from a camera\", in its "
        "embodiments mainly digital gloves that measure the movements and angles of both hands. Each sign is "
        "modeled by a hidden Markov model, and the key idea is to model the movement between signs explicitly: "
        "separate models describe the transition before the first sign of a phrase, between two signs and after "
        "the last sign. The sign and transition models are connected into a decoding network, as words and pauses "
        "are in speech recognition, and a language model gives the prior probability of each sentence, so that a "
        "whole phrase can be recognized without pauses between signs. The system treats the dominant and "
        "non-dominant hand separately; it can use only the global angles of the non-dominant hand, which allows a "
        "simpler glove on that hand, and it ties the non-dominant-hand parameters of one-handed signs to the "
        "transition model to reduce errors. The output device produces a text or audio translation of the "
        "recognized signs.")
    b.body(
        "The granted claims are narrower than this general description. Independent claim 1 requires separate "
        "inputs for the movements of the dominant and the non-dominant hand, finger-related features extracted from "
        "the dominant hand and global-angle features from the non-dominant hand, a first hidden Markov model that "
        "detects the start of a sign sequence, a second one that identifies the first sign, and an output for that "
        "sign; the dependent claims add the transition and end models. The patent was also filed in Europe, China "
        "and through the Patent Cooperation Treaty, which shows that Bosch saw commercial value in sign language "
        "recognition as a human-machine interface, not only as an assistive aid.")
    b.body(
        "**Difference from our proposal.** Bosch relies on wearable sensors, while our device needs nothing worn: "
        "the signer stands in front of one camera, which matches the preference of the Deaf users surveyed, of "
        "whom only 38% were comfortable with wearables {ref:1}. Bosch uses hidden Markov models, while we use a "
        "neural network on body and hand landmarks. Its explicit transition models solve a problem we also face, "
        "recognizing several signs made without a pause; we currently handle it differently, with a rest pose "
        "between signs and, for fluent signing, candidate cuts where the hands slow down that the recognizer then "
        "scores (Section 7.3). Bosch's system is one-way and does not describe abstention, confirmation or "
        "multilingual speech.")
    b.body(
        "**Expired patent: Hitachi, U.S. Patent No. 5,887,069** {ref:21}. Granted in 1999 to Hitachi, Ltd. with a "
        "priority date of 1992, this patent recognizes continuous sign language from data gloves or a camera by "
        "matching compressed time sequences against averaged reference patterns, and describes a translation system "
        "that also recognizes the signer's facial expression and adds the matching emotion to synthesized speech. "
        "Google Patents lists it as expired, so its methods are free to use. It shows that sign-to-speech "
        "translation has been patented for three decades; our device differs in using learned models, a single "
        "ordinary camera, abstention and confirmation, and several output languages.")

    # ---------------------------------------------------------------- 8.4 comparison
    b.heading(2, "8.4 Originality and Comparison")
    b.body(
        "Table 23 compares the reviewed patents with the proposed device. Each feature of our design exists "
        "somewhere in the prior art: a single camera in Sign-Speak and Sorenson, a two-way device between two "
        "people in Avodah, user confirmation of candidate sentences in Sign-Speak, and several output languages in "
        "Avodah. No reviewed system, however, combines them, and none decides from a calibrated confidence when to "
        "accept a sign and when to abstain. The originality of our proposal therefore lies in the combination of "
        "fully offline processing on an embedded GPU with one ordinary camera, calibrated abstention followed by "
        "confirmation of every sentence before it is spoken, and multilingual speech with a reply channel in one "
        "stand-alone device; and in its evaluation, which reports accuracy for each signer and the trade-off "
        "between answering and abstaining. The review also showed the limits of this claim: recognition from body "
        "and hand pose is itself claimed by the granted Sign-Speak patent, and a major provider of relay services "
        "is working on the same problem. These findings are reflected in the legal constraints of Section 6.2.")
    nd = "Not described"
    b.table("Comparison of the reviewed patents with the proposed device.",
            [1590, 1300, 1300, 1300, 1290, 1290, 1290],
            ["Feature", "Sign-Speak {ref:19}", "Avodah {ref:18}", "Sorenson {cite:sorenson}", "Bosch {ref:20}",
             "Hitachi {ref:21}", "Proposed device"],
            [["Input", "One camera", "Several cameras and a depth sensor", "Video from one device",
              "Gloves (camera possible)", "Data gloves or camera", "One RGB camera"],
             ["Where processing runs", "Server required", "Local, remote or both", "Phone or relay servers",
              "Local processor", "Local computer", "On the device, fully offline"],
             ["Recognition method", "Pose keypoints, nearest neighbour with time warping",
              "Neural network on a 3D composite", "Matching to stored videos with context",
              "Hidden Markov models", "Pattern matching", "Pose landmarks, neural network"],
             ["Calibrated confidence, abstains when unsure", nd, nd, nd, nd, nd, "Yes"],
             ["User confirms before output", "Yes, menu of three sentences", nd, nd, nd, nd,
              "Yes, every sentence"],
             ["Output", "Text", "Speech, text or avatar", "Voice or text", "Text or audio",
              "Text or speech with emotion", "Speech in EN, TR, ES"],
             ["Reply from the hearing person", nd, "Yes", "Yes, voice to text", nd, nd, "Yes, speech to text"],
             ["Legal status (Oct. 2026)", "Active to 2043", "Active", "Active to 2042", "Active", "Expired", "—"]],
            bold_first_col=True, shade_rows=())
