## Positional encoding homework assignments - 2.
We explore using FT (fourier transform) as filters to print a flattened image (1D signal):
```
├── results.pptx        results
├── v1_xyz              NN takes raw pixel position as input
└── v2_FT
    ├── FT_1layer       FT as positional encoding: FT encodies pixel position only
    └── FT_2layers      FT as filter: FT encodes pixel position and the output of the 1st layer to account for the 2D correlation info.
```
`print.py` is the main python code.

### Knowledge points:
1. Fourier series can be applied to any signals, thus it is not limited to positions as in the positional encoding paper. Therefore, we can also apply FT on the output of layers.
2. Applying FT on both input positions and the output of the 1st layer can take into account 2D correlation info that is missing in the flattened image.
3. The fundamental frequency of FS: f = 1/P, where P is the range/period of the signal. Choosing the fundamental frequency is important. It will ensure complete waves on P and fast converge of training.
