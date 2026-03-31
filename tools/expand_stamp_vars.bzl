def expand_stamp_vars_multi(
        name,
        templates,
        outs,
        visibility = None):

    if len(templates) != len(outs):
        fail("templates and outs must have same length")

    cmds = []
    for i in range(len(templates)):
        cmds.append("""
        $(location //tools:expand_stamp_vars) \
          $(location stable_status) \
          $(location volatile_status) \
          < $(location {src}) > {out}
        """.format(
            src = templates[i],
            out = outs[i],
        ))

    native.genrule(
        name = name,
        srcs = templates,
        outs = outs,
        tools = ["//tools:expand_stamp_vars"],
        cmd = "\n".join(cmds),
        stamp = 1,
        visibility = visibility,
    )
