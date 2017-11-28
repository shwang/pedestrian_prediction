def unpack_opt_list(iter_or_scalar, require_not_empty=True, extend_to=1):
    """
    Converts the iter to a list, or the scalar to a list.
    """
    try:
        iter(iter_or_scalar)
        res = list(iter_or_scalar)
        if require_not_empty:
            assert len(res) > 0, res
    except TypeError:
        res = [iter_or_scalar] * extend_to
    return res
