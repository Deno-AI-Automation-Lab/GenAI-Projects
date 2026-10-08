KANYAKUMARI – 2 MINUTE TAMIL DOCUMENTARY
VERSION 4

WORKFLOW:
1. Open a NEW Gemini chat for Scene 01.
2. Paste 01_scene_01.json and generate the video.
3. Download the MP4.
4. Start a NEW Gemini chat for Scene 02.
5. Repeat for all 12 scenes.
6. NEVER attach the previous scene to the next scene.
7. NEVER use Extend, Continue, Edit, Remix, or video-to-video.
8. Combine the 12 independent MP4 files only after generation.

EXPECTED:
Each scene = 10 seconds
12 scenes = 120 seconds / 2 minutes.

AUDIO:
0–1 sec: ambience/music, no speech
1–8.5 sec: one Tamil narrator
8.5–10 sec: no speech
Tamil spoken audio only. No English. No overlapping voices.

WHY VERSION 4:
Gemini may treat multiple JSONs in the same chat as instructions to extend or edit the accumulated video. That can cause unexpected durations and regenerated/removed footage. Version 4 explicitly makes every scene STANDALONE_NEW_VIDEO and separates generation from final assembly.
