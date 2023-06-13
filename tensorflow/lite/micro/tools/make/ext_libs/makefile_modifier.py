

def fill_makefile(func):
    with open("Makefile","r") as fid:
        tmp_file = fid.read()

    new_file= func(tmp_file)

    with open("Makefile","w") as out:
        out.write(new_file)



if __name__ == "__main__":
    import sys
    kernels = sys.argv[1]

    if kernels=="siliconlabs":
        from siliconlabs import update_makefile
        fill_makefile(update_makefile)

