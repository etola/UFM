#!/usr/bin/env python3
# --------------------------------------------------------
# Adopted from AnyMap
# Module containing metadata about the BlendedMVS dataset
# --------------------------------------------------------
import os


class MY_DATA_Module:
    """
    The main building block for the BlendedMVS dataset. This class contains the _information_ about the BlendedMVS dataset, and implements no functionality.
    All functionalities are implemented in inherited classes.

    Root directory structure:
    .
    └── my_data/
        ├── 000000000000000000000000/
        │   ├── All the frame data: Depth (img_name.exr), Image (img_name.jpg), Camera Params (img_name.npz)
        ├── ...
        ├── 5a2a95f032a1c655cfe3de62
        ├── ...
        ├── 5858dbcab338a62ad5001081
        └── ...
    """

    def __init__(self, data_root):
        self.data_root = data_root

        # All the 502 sequences in the dataset (totals to 115k images)
        self.all_sequences = [
            "00000000000000000000000a",
            "00000000000000000000000b",
            "00000000000000000000000d",
            "00000000000000000000000e",
            "00000000000000000000000f",
            "000000000000000000000001",
            "00000000000000000000001b",
            "00000000000000000000001d",
            "000000000000000000000002",
            "000000000000000000000003",
            "000000000000000000000004",
            "000000000000000000000005",
            "5a2a95f032a1c655cfe3de62",
            "5a2af22b32a1c655cfe46013",
            "5a2ba6de32a1c655cfe51b79",
            "5a3b9731e24cd76dad1a5f1b",
            "5a3ca9cb270f0e3f14d0eddb",
            "5a3cb4e4270f0e3f14d12f43",
            "5a03e732454a8a7ec672776c",
            "5a3f4aba5889373fbbc5d3b5",
            "5a4a38dad38c8a075495b5d2",
            "5a5a1e48d62c7a12d5d00e47",
            "5a6b1c418d100c2f8fdc4411",
            "5a6feeb54a7fbc3f874f9db7",
            "5a7cb1d6fe5c0d6fb53e64fb",
            "5a7d3db14989e929563eb153",
            "5a8aa0fab18050187cbe060e",
            "5a9e5df65baeef72b4a021cd",
            "5a48ba95c7dab83a7d7b44ed",
            "5a48c4e9c7dab83a7d7b5cc7",
            "5a48d4b2c7dab83a7d7b9851",
            "5a69c47d0d5d0a7f3b2e9752",
            "5a77b46b318efe6c6736e68a",
            "5a355c271b63f53d5970f362",
            "5a489fb1c7dab83a7d7b1070",
            "5a533e8034d7582116e34209",
            "5a562fc7425d0f5186314725",
            "5a572fd9fc597b0478a81d14",
            "5a588a8193ac3d233f77fbca",
        ]

        # Final sequences to be used after filtering (some of the sequences have incorrect/low quality depth)
        # Generally water bodies like lakes have incorrect depth
        # Filtered out sequences:
        # '5692a4c2adafac1f14201821' # Incorrect Depth
        # '5864a935712e2761469111b4' # Noisy Depth and artifacts near horizon
        # '59f87d0bfa6280566fb38c9a' # Object-centric, noise with background and sometimes in front of object
        # '58a44463156b87103d3ed45e' # Very noisy depth in background
        # '5c2b3ed5e611832e8aed46bf' # Depth occluded by artifacts
        # '5bf03590d4392319481971dc' # Depth occluded by artifacts
        # '00000000000000000000001a' # Largely incomplete depth
        # '00000000000000000000000c' # Imprecise depth for buildings
        # '000000000000000000000000' # Incorrect depth for planar terrain
        self.sequences = [
            "00000000000000000000000a",
            "00000000000000000000000b",
            "00000000000000000000000d",
            "00000000000000000000000e",
            "00000000000000000000000f",
            "000000000000000000000001",
            "00000000000000000000001b",
            "00000000000000000000001d",
            "000000000000000000000002",
            "000000000000000000000003",
            "000000000000000000000004",
            "000000000000000000000005",
            "5a2a95f032a1c655cfe3de62",
            "5a2af22b32a1c655cfe46013",
            "5a2ba6de32a1c655cfe51b79",
            "5a3b9731e24cd76dad1a5f1b",
            "5a3ca9cb270f0e3f14d0eddb",
            "5a3cb4e4270f0e3f14d12f43",
            "5a03e732454a8a7ec672776c",
            "5a3f4aba5889373fbbc5d3b5",
            "5a4a38dad38c8a075495b5d2",
            "5a5a1e48d62c7a12d5d00e47",
            "5a6b1c418d100c2f8fdc4411",
            "5a6feeb54a7fbc3f874f9db7",
            "5a7cb1d6fe5c0d6fb53e64fb",
            "5a7d3db14989e929563eb153",
            "5a8aa0fab18050187cbe060e",
            "5a9e5df65baeef72b4a021cd",
            "5a48ba95c7dab83a7d7b44ed",
            "5a48c4e9c7dab83a7d7b5cc7",
            "5a48d4b2c7dab83a7d7b9851",
            "5a69c47d0d5d0a7f3b2e9752",
            "5a77b46b318efe6c6736e68a",
            "5a355c271b63f53d5970f362",
            "5a489fb1c7dab83a7d7b1070",
            "5a533e8034d7582116e34209",
            "5a562fc7425d0f5186314725",
            "5a572fd9fc597b0478a81d14",
            "5a588a8193ac3d233f77fbca",
        ]
