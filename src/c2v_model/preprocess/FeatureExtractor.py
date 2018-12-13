import os
import sys
import subprocess
from subprocess import Popen, PIPE, STDOUT, call
from threading import Timer

class Extractor(object):
    @staticmethod
    def extract(file_path, path_to_extractor="c2v_model/preprocess/Extractor.jar", max_path_length="8", max_path_width="2"):
        command = ["java", "-cp", path_to_extractor, "JavaExtractor.App", "--max_path_length", max_path_length,
                   "--max_path_width", max_path_width, "--file", file_path, "--no_hash"]
        # kill = lambda process: process.kill()
        # Now start extracting
        sleeper = subprocess.Popen(command, stdout=PIPE, stderr=PIPE)
        out = sleeper.stdout.read()
        err = sleeper.stderr.read()
        # timer = Timer(1.0, kill, [sleeper])
        # try:
            # timer.start()
            # out = sleeper.stdout.read()
            # err = sleeper.stderr.read()

        # finally:
            # timer.cancel()
        
        # if sleeper.poll() == 0:
        #     # Now let's deal with errors
        #     if len(err) > 0:
        #         print(err, file=sys.stderr)
        #     else:
        #         print("File Extracting Time Out: " + file_path, file=sys.stderr)
        return out, err

    @staticmethod
    def java_string_hashcode(s):
        """
        Imitating Java's String#hashCode, because the model is trained on hashed paths but we wish to
        Present the path attention on un-hashed paths.
        """
        h = 0
        for c in s:
            h = (31 * h + ord(c)) & 0xFFFFFFFF
        return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000

'''
if __name__ == "__main__":
    out, _ = Extractor.extract("../Test.java")
    print(out.decode())
'''