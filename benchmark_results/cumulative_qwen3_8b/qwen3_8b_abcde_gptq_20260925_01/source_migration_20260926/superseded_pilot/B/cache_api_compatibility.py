def cache_mask_sizes(self, cache_position, layer_idx):
    """Mask metadata for the existing paged cache under Transformers 4.57."""
    return self.length + cache_position.shape[0], 0
