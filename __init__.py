from .custom_nodes import JigsawPuzzleEffect, RegionBoundaryEffect

NODE_CLASS_MAPPINGS = {
    "MyJigsawPuzzleEffect": JigsawPuzzleEffect,
    "MyRegionBoundaryEffect": RegionBoundaryEffect
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyJigsawPuzzleEffect": "My Jigsaw Puzzle Effect",
    "MyRegionBoundaryEffect": "My Region Boundary Effect"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']