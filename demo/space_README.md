---
title: PariVision traffic events demo
emoji: 🚦
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
short_description: Traffic event detection API for the WIUT Hackathon 2026 demo
---

Backend for the live demo on our team site. It runs the same pipeline as the
submission (YOLO26 + ByteTrack + rules on trajectories, causal risk model) on
the CPU, on the first 120 s of an uploaded clip, and returns events, a risk
curve and an annotated video. Source: see the repository linked from the site.
