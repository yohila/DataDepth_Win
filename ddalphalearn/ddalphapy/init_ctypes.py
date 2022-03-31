def __init__(self):
            try:
                elif platform.system() == 'Windows':
                    if sys.version_info.major == 3 and sys.version_info.minor == 8:
                        import os
                        os.add_dll_directory(os.getenv("dir", default=r'C:\Users\ayamoul\Downloads\ddalpha_create5\ddalpha_create4\ddalphalearn\ddalphacpp'))
                    self.c_khiva_library = ctypes.CDLL('./ddalpha_package.dll')

            except:
                raise Exception("Khiva C++ library is required in order to use the Python Khiva library")