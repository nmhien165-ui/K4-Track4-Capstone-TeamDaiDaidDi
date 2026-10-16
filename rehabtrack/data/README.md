# Public datasets: reference only (checked 2026-10-16)

## Purpose

These datasets are listed as reference only. None of them is a paired
pre-operative / post-operative outcome set, and none substitutes for the two
staged side-view clips the team must record (same participant, same camera,
same seated knee extension, before vs after). They are here to show what exists,
what is gated, and why the staged clips are still required. Nothing from these
datasets has been downloaded; every entry below comes from the project page,
README, Zenodo record, or paper page read on 2026-10-16. Anything not seen on
those pages is marked UNVERIFIED.

## Datasets checked

| Dataset | What is verified | RGB video? | Seated knee extension? | Access / license | Usable for this MVP now? | Source URL | Date checked |
|---|---|---|---|---|---|---|---|
| UCO Physical Rehabilitation (UCOPhyRehab) | 27 subjects (7 F, 20 M), age 23-60, recorded Jun-Jul 2022 in a lab. 16 activities (8 lower body, 8 upper body, left and right), 4 repetitions each, slow speed "to simulate a recent surgery or injury". 2160 video sequences, mean 30.4 s, 5 RGB cameras at 1280x720, 3 horizontal angles x 3 heights (cameras 0-2 share the horizontal angle; camera 0 at 15 cm, camera 1 at 180 cm, others at 100 cm). OptiTrack 3D ground truth with markers at hip, knee, ankle (lower body), 2D points by projection. | Yes (camX.avi per camera) | Closest match: exercises 01/05 "Bending the knee without support while sitting" (seated, raise leg as high as possible); 02/06 same with support of the other leg. Left and right sides. | Email request to inforeha@uco.es with name, affiliation and research purpose (README). License / terms: UNVERIFIED (none shown on the README). Which camera is a true side view: UNVERIFIED. Subjects appear to be volunteers, not patients: UNVERIFIED. | Only as a gated, optional pose-pipeline check after an email approval. No for the MVP evaluation (not paired pre/post, access gate). | https://github.com/AVAuco/ucophyrehab | 2026-10-16 |
| UCOPhyRehab++ | Zenodo record published 2025-12-15, v1, last updated 2026-05-19, 9.7 GB total. Contains derived modalities only: silhouettes, cropped silhouettes, optical flow (GMFlow and TV-L1), semantic segmentation, plus `dataset_3d_with_angles.json` (230 MB, OptiTrack 3D joints and derived joint angles) and a metadata `.jsonl` (149 kB). `samples.zip` (326.8 MB) holds one subject with all modalities. Same subject / exercise / camera layout as the original set. | No. The record states the modalities are "computed from the original RGB recordings"; the original RGB is not in this release. | Not stated on the record (same exercise protocol as the original set is implied but UNVERIFIED; the record has no exercise list). | Open download, no request form. License CC-BY-4.0 (Zenodo API field). | No for YOLO pose (no RGB frames). Only as a source of ground-truth knee angles (3D JSON) for a sanity comparison, and only if the exercise labels match (UNVERIFIED). | https://zenodo.org/records/17935737 | 2026-10-16 |
| KneE-PAD | 31 patients with knee pathologies selected from 267 monitored over 6 months in 2 physiotherapy centers. Exercises: squat, seated leg extension, gait (TUG protocol), each with 2 wrong variations (9 classes). 2,086 files of about 4.2 s. Data are 8 Delsys sEMG + IMU sensors saved as `.npy`. Zenodo total 289.4 MB. The paper states an RGB camera, goniometer, heart-rate and muscle-oxygenation sensors were used for annotation only and "are not provided within KneE-PAD". | No (recorded, not distributed) | Yes, as a class (Extension, Extension_NF, Extension_LL), but only as sEMG/IMU signals, no video and no goniometer angle | Open download. License CC-BY-4.0 (Zenodo API field; the paper is also CC-BY 4.0). | No. No video and no knee angle are distributed. | https://github.com/ounospanas/KneE-PAD (data: https://zenodo.org/records/12112951; paper: https://www.nature.com/articles/s41597-025-04963-4) | 2026-10-16 |
| KERAAL | Low back-pain rehabilitation dataset: 9 healthy subjects and 12 patients, 3 exercises (hiding face CTK, flank stretch ELK, torso rotation RTK), recorded during robot coaching. RGB video (960x544 avi for group 3; 480x360 mp4 for groups 1A, 1B, 2A, 2B), Kinect skeleton, Vicon skeleton, OpenPose 2D and BlazePose 3D skeletons, plus therapist annotations of correctness and errors. Group sizes listed on the page: 1A 1.4 GB, 1B 13 GB, 2A 338 MB, 2B 1.3 GB, 3 8.2 GB. | Yes | No (torso / upper-body exercises) | License CC-BY-NC-SA. The page offers a sample zip and per-group downloads; I did not open the links, so whether they need a form or login is UNVERIFIED. | No. Wrong exercise and wrong body region. Could only serve as a generic RGB-video pose pipeline check. | https://keraal.enstb.org/KeraalDataset.html | 2026-10-16 |
| UI-PRMD | 10 healthy subjects x 10 movements x 10 repetitions, captured with a Vicon optical tracker and a Microsoft Kinect. Data are positions and angles of full-body joints. Movements: deep squat, hurdle step, inline lunge, side lunge, sit to stand, standing active straight leg raise, and four standing shoulder movements. | No RGB video is described in the paper text read; raw video availability: UNVERIFIED (distributed data are described as joint positions and angles) | No. Closest is sit to stand and standing straight leg raise; neither is a seated knee extension. | License ODC Public Domain Dedication (PDDL v1.0). Download location / form: UNVERIFIED (not stated in the paper text read). | No. Skeleton and angle data only, healthy subjects, wrong exercise. | https://doi.org/10.3390/data3010002 | 2026-10-16 |

## Verdict: are public datasets sufficient for the MVP evaluation?

No. They can inform the design, but they cannot replace the staged clips.

1. **Pairing.** The MVP question is whether the same person's knee angle changes between two sessions (before vs after). Every dataset above records a single session per person. None provides paired pre/post-operative recordings, so none can test "did the pipeline detect the change".
2. **Same participant, same camera.** The 2D knee angle is only comparable between two clips when the person, camera position and framing are the same. Public sets give different people per recording, or several cameras at different heights and angles, so a before/after pair cannot be built from them honestly.
3. **Access gates.** The only dataset here with seated-knee RGB video that the pipeline can use directly (UCOPhyRehab) is released by email request with a stated research purpose; its license is not shown. UCOPhyRehab++ is open but has no RGB frames. KneE-PAD is open but has no video. Waiting on an email approval does not fit a hackathon timeline.
4. **Exercise mismatch.** KERAAL is back-pain torso work, UI-PRMD is squat / lunge / shoulder work, KneE-PAD has the right exercise but only as wearable signals. Only UCOPhyRehab lists a seated knee movement on video.

What they can still do: UCOPhyRehab (after approval) could serve as an optional extra check that the frozen YOLO pose model finds the hip, knee and ankle on a seated leg raise, and UCOPhyRehab++ could supply ground-truth knee angles. Neither is required for the demo.

## Optional smoke-test clip candidates (pose extraction only)

Use only to confirm the pipeline runs (YOLO pose finds hip, knee, ankle; angle code returns numbers). Not a paired before/after session and not evidence for the evaluation. Nothing was downloaded. I did not play either clip, so the camera angle (side view or not) is UNVERIFIED for both; check the preview before use.

| Clip | What the page says | Size / duration | License | URL |
|---|---|---|---|---|
| Pexels, NoomaN YolchU | "An adult man exercises on a leg extension machine in a modern gym." | 3840x2160, 0:48, 30 fps (file size UNVERIFIED; pick a lower resolution at download) | Pexels license: free to use, attribution not required, modification allowed (this is the Pexels license, not CC0). Note it bars implying endorsement and showing identifiable people in a bad light. | https://www.pexels.com/video/focused-workout-with-leg-extension-machine-36539451/ |
| Pexels, Edu Maciel | "Adult male in gym performing leg extensions" | 2160x3840 portrait, 0:09, 30 fps (file size UNVERIFIED) | Pexels license (same terms) | https://www.pexels.com/video/gym-workout-motivation-with-leg-extensions-32709558/ |

Both are gym leg-extension-machine style clips rather than a seated chair knee extension, so the movement is only approximately the target. No clearly side-view, chair-based, openly licensed clip was confirmed.

## Not used

RSNA knee MRI competition data: wrong modality (MRI) and wrong task (image classification, not video pose / joint angle).
