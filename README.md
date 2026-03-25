# Dual-Plane World Model (DPWM)

Prototype implementation for the paper:
"Dual-Plane World Model: Reversible Composition 
of Physical and Linguistic Embeddings"

## Install
pip install torch numpy

## Run
python dual_plane_prototype.py

## Key idea
combined = physics_vec + lang_vec
recovered_physics = decoder_p(combined - lang_vec)
recovered_lang = decoder_l(combined - physics_vec)

## Result
Physics recovery error: 0.0109 ✓

## Author
Artem Gorbunov, Independent Researcher, 2026
