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

def StreamBurning(filedem, fileriver, fileburn="", value=20, verbose=False):
    """
    StreamBurning
    """
    fileburn = fileburn if fileburn else forceext(filedem, "brn.tif")
    env = {
        "burn": fileburn,
        "river": fileriver,
        "dem": filedem,
        "value": value
    }
    return gdal_numpy("""burn=np.where(river>0,dem-value,0)""", env, verbose=verbose)

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

def D8FlowDir(felfile, pfile="", sd8file="", n=-1, skipIfExists=False, verbose=False):
    """
    D8FlowDir
    """
    command = """{exe} -fel "{felfile}" -p "{pfile}" -sd8 "{sd8file}" """

    pfile = remove_suffix(felfile, "fel") + "p.tif" if not pfile   else pfile
    sd8file = remove_suffix(felfile, "fel") + "sd8.tif" if not sd8file else sd8file
    mkdirs(justpath(pfile))
    mkdirs(justpath(sd8file))

    env = {"exe": "d8flowdir", "felfile": felfile, "pfile": pfile, "sd8file": sd8file, "n": n}
    return mpiexec(command, env, n, precond=[felfile], postcond=[pfile, sd8file], skipIfExists=skipIfExists,
                   verbose=verbose)

def AreaD8(pfile, ad8file="", nc=True, n=-1, skipIfExists=False, verbose=False):
    """
    AreaD8
    """
    command = """{exe} -p "{pfile}" -ad8 "{ad8file}" {nc} """

    ad8file = remove_suffix(pfile, "p") + "ad8.tif" if not ad8file else ad8file
    mkdirs(justpath(ad8file))

    env = {"exe": "aread8", "pfile": pfile, "ad8file": ad8file, "nc": "-nc" if nc else "", "n": n}
    return mpiexec(command, env, n, precond=[pfile], postcond=[ad8file], skipIfExists=skipIfExists, verbose=verbose)
