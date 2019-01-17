# -------------------------------------------------------------------------------
# Licence:
# Copyright (c) 2012-2019 Luzzi Valerio 
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
#
# Name:        taudem.py
# Purpose:
#
# Author:      Luzzi Valerio
#
# Created:     17/01/2019
# -------------------------------------------------------------------------------
import os,sys
from gecosistema_core import *

def mpiexec(command, env={}, n=0, precond=[], postcond=[], remove=[], skipIfExists=False, verbose=False):
    """
    mpiexec
    """
    n = n if n else cpu_count()

    if isWindows():
        if Exec("mpiexec"):
            env["__mpiexe__"] = "mpiexec"
        # elif os.environ.has_key("MSMPI_BIN"):  #Python2
        elif "MSMPI_BIN" in os.environ:
            env["__mpiexe__"] = '"' + os.environ["MSMPI_BIN"] + "\\mpiexec.exe" +'"'
        else:
            print("Warning: may be mpiexec is not installed!")
            n = 1

    env["__n__"] = n
    if n > 1:
        command = """mpiexec -n {__n__} """ + command
    if Exec(command, env, precond, postcond, remove, skipIfExists, verbose=verbose):
        return postcond[0] if len(postcond) == 1 else tuple(postcond)
    return False

def PitRemove(demfile, felfile="", n=-1, skipIfExists=False, verbose=False):
    """
    PitRemove
    """
    command = """{pitremove} -z "{demfile}" -fel "{felfile}" """

    felfile = name_without_ext(demfile) + "fel.tif" if not felfile else felfile
    mkdirs(justpath(felfile))
    env = {"pitremove": "pitremove", "demfile": demfile, "felfile": felfile, "n": n}

    felfile = mpiexec(command, env, n, precond=[demfile], postcond=[felfile], skipIfExists=skipIfExists,
                      verbose=verbose)

    # Fix nodata if needed
    if felfile and GetNoData(felfile) != GetNoData(demfile):
        nodata = GetNoData(demfile)
        if verbose:
            print("fixing nodata with %s" % nodata)
        gdal_numpy("f=np.where(abs(f)>1e+38,{nodata},f)", {"f": felfile, "nodata": nodata})
        SetNoData(felfile, nodata)

    return felfile