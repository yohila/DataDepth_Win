from setuptools.command.build_ext import build_ext
from setuptools import Extension, setup, find_packages


class custom_build_ext(build_ext):
    def build_extensions(self):
        # Override the compiler executables. Importantly, this
        # removes the "default" compiler flags that would
        # otherwise get passed on to to the compiler, i.e.,
        # distutils.sysconfig.get_var("CFLAGS").
        self.compiler.set_executable("compiler_so", "g++ -c -fpic")
        self.compiler.set_executable("compiler_cxx", "g++")
        self.compiler.set_executable("linker_so", "g++ -Wl,--gc-sections -shared -lstdc++")
        build_ext.build_extensions(self)


setup(
    name="ddalphalearn",
    packages=find_packages(),
    ext_modules=[
        Extension(
            "ddalphacpp", 
            sources=["ddalphalearn/ddalphacpp/ddalpha.cpp"],
            extra_compile_args=["-I."]
            
        )
    ],
    zip_safe=False,
    cmdclass={"build_ext": custom_build_ext}
)
