"""
MIFCraft: MIF File Metaprogrammer Copyright (C) 2012 Mark Mascaro

This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation, either version 3 of the License, or (at your option) any later 
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY 
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program, in the readme PDF. If not, see <http://www.gnu.org/licenses/>.

Also add information on how to contact you by electronic and paper mail.
Single-file module providing a scripting approach to creating MIF files for
the OOMMF micromagnetic simulator. Provides input validation and restores
control flow constructs for rapid production of simulations with varying
parameters. 

Provides the MIF class, which must be instantiated with the keyword arguments
filename (the MIF file to be written) and basename (the OOMMF basename prefix).
The MIF class supports Python 'with' semantics used for error trapping and 
validation of the final MIF file; please use 'with'.

All Oxs_Foo Specify blocks are provided as Foo methods on the MIF class. All
such methods take only keyword arguments, which are named the same as in the 
relevant Specify block in the OOMMF specification. Tuples and lists are used
in sane places where they would be intuitive; consult the accompanying 
reference manual to clear up any confusion.

Usage Example:

import mifcraft

with mifcraft.MIF(filename='foo.mif', basename='foo') as M:
  M.BoxAtlas(xrange = (0, 1e-8),
             yrange = (0, 1e-8),
             zrange = (0, 5e-9))
  ...
  
Full examples are provided in the normal package download.
"""

from __future__ import division
from __future__ import print_function
    
from functools import wraps
from collections import defaultdict
import os, sys, datetime

"""
If you're still reading, you're probably a programmer examining or working on 
MIFCraft. In that case, a few design decisions need to be explained:

MIFCraft is implemented as one enormous MIF class with many, many methods. Of 
course, 95% of them do just one sort of thing: write a Specifyblock to the 
output file. Any file-writing function needs a reference to the MIF object, so 
the choices are passing the MIF object or making them all methods. MIFCraft uses 
methods because it preserves the keyword-arg-only aspect of the 
Specify-replacement functions. 

Keyword-arg-only functions are used for two reasons

1> MIFCraft should be symmetric wherever possible with the OOMMF MIF 
specification, so that users feel immediately familiar. 
  a> Using only keyword arguments creates high symmetry: OOMMF users are used 
     to supplying "label value" pairs as arguments. 
  b> MIF arguments are order-agnostic; keyword arguments preserve that property. 
  c> The MIF object itself is nothing like the Specify function arguments; it is
     intuitive to a naive user that this piece of apparent boilerplate goes on 
     the 'outside' of the function.
  d> It makes it very unlikely that the user will arbitrarily decide to use an
     argument without a label (which generates a TypeError that cannot be as
     informative at other errors). There's simply no context for arguments
     without labels anywhere in MIFCraft.

2> The MIFWrite wrapper function, in particular, takes *args and **kwargs,
passing down **kwargs. This means the user can supply pretty much any set of
valid python expressions and make it inside the MIFWrite decorator body. Once 
inside the decorator body, errors will be caught by the MIF exception handler, 
which generates simple error messages for the user. This structure ensures 
there are very few ways to generate an exception at a high enough level 
that the user is stuck reading a backtrace. Prevents terror.
"""


#############
# Utilities #
#############

def correctPathString(pathString):
    """Replace Windows and Linux path separators with locally correct version."""
    return pathString.replace("/", os.path.sep).replace("\\", os.path.sep)

def carefulFloatMod(x, y):
    """
    Does some fudging around floating points to make modulo work for this purpose.
    Returns True if there is a significant remainder.
    
    Don't look at me, the OOMMF spec relies on floating point modulo.
    """
    def sign(f):
        return f/abs(f)
    return abs((x + 1E-16*sign(x)) % y) > 5E-16

# You want to read about the exception structure before reading this function.
def quickValidateExtent(brick, name, xmin, xmax, ymin, ymax, zmin, zmax):
    """
    Takes a calling Specify block type and name, and then 6 coordinates
    x, x', y, y', z, z'. Makes sure all coordinates are well-ordered as
    OOMMF expects; raises otherwise.
    """
    for label, nmin, nmax in [("x", xmin, xmax), ("y", ymin, ymax), ("z", zmin, zmax)]:
        if nmin > nmax:
            # In this case, we're in a helper function and can't rely
            # on exception autodetection to get the right calling context.
            # Pass explicitly.
            raise MIFException(label + "min, " + label + "max", 
                               "min %s is greater than max %s" % (nmin, nmax),
                               brick, name)

def sigfigs(lst):
    """
    Takes a list of numerical arguments. Determines the minimum number of
    significant figures needed to properly represent ALL arguments. Typical
    use: formatting all numbers to print the same width.
    
    Returns an int indicating the correct number of places after the
    decimal point.
    """
    # Fortunately, str(aNumber) will use as much space as necessary.
    # That means we can check lengths to get sigfigs.
    
    # We don't want the whole length, though:
    # Six is a magic number. It's how much space is wasted by the decimal 
    # point, the E, etc - how much space we *don't* count when checking the 
    # length of the string representation.
    usedLen = map(lambda x: len(str(abs(x)))-6, lst)
    # We're interested in the longest representation, but for no reason should
    # we EVER use fewer than two significant figures. It's the indicator that 
    # we have a float, in a sense.
    return max(max(usedLen), 2)

#############################
# Decorators and Exceptions #
#############################

def MIFWrite(fn):
    """Wraps MIF methods that create Specify blocks to write them to files.
    
    Wraps a MIF method that returns a list of strings, generally a Specify
    block. Writes the strings  ('\n'-joined) safely to the MIF object's 
    specified file to avoid repetitive open/close calls. If no name for a 
    Specify block is supplied in the 'name' kwarg, a unique name is generated
    based on the block type and a LUID local to this MIF object. The LUID is
    then incremented.
    
    Must always wrap a MIF instance method, as it relies on getting self as the
    first positional arg. 
    
    Returns the generated (or supplied) name, squelching whatever the wrapped
    function was going to return after writing is done.
     
    The kwarg '_noWrite' can be specified True. If this is done, no file writing 
    or UID incrementing occurs, and the string is returned instead. This is used
    in the very complex Evolvers to allow some code reuse. This should be used 
    only with extreme caution.
    """
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        ## Step 1: Deal with naming
        if kwargs.get("_noName"):
            name = None
        else:
            # Name must be supplied or generated
            if "name" in kwargs:
                name = kwargs["name"]
                if name in self.reservedLabels:
                    # User manually passed a bad name; blame them.
                    raise MIFException("name", "Block name %s already in use" % name)
            else:
                # Infer name from calling function name, which looks like
                # Specify block name
                name = fn.__name__
                postfix = 2
                while name in self.reservedLabels:
                    name = fn.__name__ + "_" + str(postfix)
                    postfix += 1
            # Invalidate future use of name in this MIF object
            self.reservedLabels.add(name)                        
        kwargs.pop("name", None)
        
        ## Step 2: Deal with writing
        if not kwargs.get("_noWrite"):
            # Never deal with writing if something has already failed
            if self.hasFailed:
                return
            
            # Otherwise, try writing
            try:
                writeString = "\n".join(fn(self, name, *args, **kwargs))
                # If the write doesn't raise, commit it to the file
                with open(self.filename, 'a') as workFile:
                    workFile.write(writeString)
                    if kwargs.get("_noPadding"):
                        workFile.write("\n")
                    else:
                        workFile.write("\n\n\n")
            except MIFException as e:
                # Something nonspecifically bad has happened to the MIF file;
                # it caught an error and things the file cannot be completed.
                # Erase the bad file and skip this and all future writes
                self.purgeFile()
                raise
            except KeyError as e:
                # This indicates an attempt to pull arguments from kwargs failed.
                # User failed to supply a mandatory argument, then.
                # Fortunately, using kwargs lets us catch this *inside* the
                # function as a KeyError, so we can generate simple
                # errors.
                self.purgeFile()
                raise MIFException(str(e)[1:-1], "Mandatory argument %s not provided" % str(e),
                                   fn.__name__)
            except TypeError as e:
                self.purgeFile()
                # Certainly due to passing something not indexable to a list-type parameter
                # Inspect the stack and try to figure out what on earth user did.
                import traceback
                print("Couldn't evaluate block:")
                depth = 2
                traceback.print_tb(sys.exc_info()[depth], 1, file=sys.stdout)
                raise MIFException("arguments", 
                                   "Bad argument. Probably passed a single argument where a list was expected.")
            except IndexError as e:
                self.purgeFile()
                # Someone didn't pass a long enough list...
                import traceback
                print("Couldn't evaluate block:")
                traceback.print_tb(sys.exc_info()[2], 1, file=sys.stdout)
                raise MIFException("arguments", 
                                   "Passed a malformed list. Please check argument typing requirements.")
            # Return the function name only, to use as arguments elsewhere
            return name
        else:
            return fn(self, name, *args, **kwargs)
        
    return wrapped

def NonBlock(fn):
    """
    Decorates a function already decorated by MIFWrite. Skips name generation 
    and validation, which is useful for things that aren't Specify blocks, 
    hence the name.
    """
    @wraps(fn)
    def wrapped(*args, **kwargs):
        fn(*args, _noName = True, **kwargs)
        return None
    return wrapped

def NoPad(fn):
    """
    Decorates a function already decorated by MIFWrite. Alters the decoration 
    to prevent padding of the file with newlines after writing.
    """
    @wraps(fn)
    def wrapped(*args, **kwargs):
        return fn(*args, _noPadding = True, **kwargs)
    return wrapped

class MIFException(Exception):
    """
    Raised when a MIF method gets bad arguments. 
    
    Attributes:
      callingBlock -- Type of Oxs block being tried
              name -- Name of Oxs block being tried
         parameter -- Name of parameter that is invalid
             error -- String description of invalid error
             
    Initialized with the following arguments:
        parameter: name of variable causing the exception to be thrown
            error: description of the reason for the error
     callingBlock: if known, the type of the Specify block that's raising.
                   If not supplied, determined by stack inspection
      callingName: if known, the name of the Specify block that's raising.
                   If not supplied, determined by stack inspection.
    """
    def __init__(self, parameter, error, callingBlock = None, callingName = None):
        import inspect # Let's just hide this here, it's scary.    
        if callingBlock:    
            self.callingBlock = callingBlock
        else: 
            # This pulls the name of the function that raised, causing
            # this exception to be constructed.
            self.callingBlock = inspect.currentframe().f_back.f_code.co_name
        if callingName:
            self.name = callingName
        else:
            try:
                self.name = inspect.currentframe().f_back.f_locals["name"]
            except KeyError:
                self.name = "Unidentified Object"
        self.parameter = parameter
        self.error = error
        
    def __str__(self):
        return 'In %s <%s>: %s invalid: %s' % (self.callingBlock, self.name,
                                               self.parameter, self.error)
        
def UndefTokenMIFException(target, category, validOptions, 
                           callingBlock = None, callingName = None):
    """
    Convenience function that generates a MIFException for the common case
    where you're given the name of a Specify block, but you have no idea
    what it is. Probably due to a user typo. In a pinch, can be used for
    exceptions when the user supplies a string arg that isn't on a certain
    acceptable list, too.
    
    Args:
           target: name of variable causing the exception to be thrown.
                   Looked up in frame; passing the name allows report of both
                   name and value!
         category: what were trying to look up, such as Atlas or VectorField
     validOptions: values for the variable that would have been acceptable
     callingBlock: if known, the type of the Specify block that's raising.
                   If not supplied, determined by stack inspection
      callingName: if known, the name of the Specify block that's raising.
                   If not supplied, determined by stack inspection.
    """
    import inspect
    callingBlock = inspect.currentframe().f_back.f_code.co_name
    if not callingName:
        try:
            callingName = inspect.currentframe().f_back.f_locals["name"]
        except KeyError:
            callingName = "Unidentified Object"
    # Gets the value of the variable whose name was supplied as the
    # target argument. For example, if 'foo' was passed, the value of the local
    # variable foo in the calling frame is returned.
    tval = inspect.currentframe().f_back.f_locals[target]
    return MIFException(target,
                        "'%s' is not a known %s in %s" % (tval, category, 
                                                          validOptions),
                        callingBlock, callingName)   
        
###################
# Main MIF Object #
###################

class MIF(object):
    """
    An object that manages a writable MIF file and accepts commands to add to 
    that file. It also tracks used labels to catch typos and forgotten labels 
    at input validation time.
    """
    
    ##################
    # 'with' support #
    ##################
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        # User is done writing file. We need to make sure the file is OK,
        # and clean up. First step is to make sure we're exiting cleanly,
        # not due to an exception.
        if isinstance(value, MIFException):
            # Tell the user why, but don't reraise - we've done what we need to do
            print("(%s) could not be completed:" % self.filename)
            print(value)
            return True
        # This is also the only time we can catch a NameError and still hide
        # a backtrace from the user - let's do it.
        # For some reason, we're getting junk in NameError here...
        if type is NameError:
            # sys.exc_info doesn't capture the right info here,
            # but we can provide something limited.
            print("(%s) could not be completed:" % self.filename)
            print(value)
            print("Do you have a typo somewhere?")
            return True
        self.validate()
    
    ##############
    # File Setup #
    ##############
    
    def __init__(self, **kwargs):
        """
        Create a MIF object to take contents commands. Create the associated 
        file object, write the header, and ensure that all paths given are 
        valid. Create non-existent paths.
        """
        
        # Input validity check
        if not "filename" in kwargs:
            print("ERROR: Keyword 'filename' not specified for MIF object. Args supplied:")
            print(kwargs)
            return
        
        # Set up path and filename
        # Note - it's assumed that people trade mifcraft specification files 
        # around, and so work is always done to enforce the locally correct path
        # separator.
        if "path" in kwargs:
            path = correctPathString(kwargs["path"])
            filename = kwargs["filename"]
        else:
            if "\\" in kwargs["filename"]:
                prepath = kwargs["filename"].rsplit("\\", 1)
                path = correctPathString(prepath[0])
                filename = prepath[1]
            elif "/" in kwargs["filename"]:
                prepath = kwargs["filename"].rsplit("/", 1)
                path = correctPathString(prepath[0])
                filename = prepath[1]
            else: # It didn't actually split
                path = "."
                filename = kwargs["filename"]
        
        # Create directories
        if not path == ".":
            try:
                os.makedirs(path)
            except os.error:
                # This is explicitly allowed to pass silently - whether we're 
                # loading everything into one subdirectory or clobbering 
                # existing specific subdirectories, this generally happens for 
                # good reasons.
                pass
        
        # Set up basename - the name OOMMF uses as a prefix to write output
        # files. If necessary, set up output directories as well.
        if "basename" in kwargs:
            # Internally, OOMMF always separates with '/'
            self.basename = kwargs["basename"].replace("\\","/")
            # We may need to make directories for the basename, lest we see an 
            # error when oommf tries to write there
            if "/" in self.basename:
                if self.basename[0] == "/" or self.basename[1] == ":":
                    #Absolute path
                    try:
                        os.makedirs(self.basename.rsplit("/",1)[0])
                    except os.error:
                        pass
                else: #Relative path
                    try:
                        os.makedirs((path + os.path.sep + 
                                     self.basename).rsplit("/", 1)[0])
                    except os.error:
                        pass
        else:
            self.basename = filename[:-4] # ...but remove .mif
        
        # Write MIF2.1 header
        self.filename = path + os.path.sep + filename
        self.OOMMFStylePath = path.replace("\\", "/")
        with open(self.filename, "w") as f:
            f.write("# MIF 2.1\n" +
                    "# Auto-generated by MIFCraft at {0:%H:%M:%S} on {0:%A %b %d %Y}\n\n".format(datetime.datetime.today()))
            
        # Keep track of labels used - we can catch and print obvious errors
        # in a human-readable way, avoiding the verbosity of oommf errors
        
        # Keeps track of atlases, fields, etc. to see if all have been used 
        # once defined. Used to warn if a label goes unused.
        self.labelsUsed = set() 
        self.reservedLabels = set() # Tracks of names taken by Specify blocks
        
        # Now on to the OOMMF data types. Ordered and unordered data types
        # are freely mixed as appropriate for the stored value.
        self.atlases = set() # Strings, just names
        # Atlas coordinates, in lists of three 2-tuples. Used to validate
        # extent of the world, divisibility into mesh, etc.
        self.atlasCoords = {}
        self.regions = defaultdict(set) # Regions in atlases, by atlas name
        self.vectorFields = set() # names
        self.scalarFields = set() # same
        self.evolvers = [] # names, but with ordering so we can default to 1st
        self.meshes = [] # same
        self.drivers = [] # same
        self.scripts = set() # names
        self.stageCounts = set() # same
        # Filenames requested in blocks. We can check that all references files
        # exist, and can warn otherwise.
        self.referencedFilenames = set() 
        self.destinations = set() # names
        self.scheduled = False # has any output been declared?

        # Keep track of whether anything has raised a MIFException, meaning we 
        # should abort
        self.hasFailed = False
        
    
    #################################
    # Validation of Completed Files #
    #################################
    
    def validate(self):
        """Ensure file is valid and close handle; call after clean __exit__.
        
        Checks that all necessary values have been defined, all labels
        declared have been used, that the simulation has some output, etc.
        Will raise errors or print warnings accordingly.
        """
        if self.hasFailed:
            # We're already wrong, so just leave
            return
        
        # Check that objects critical to OOMMF (and not checked elsewhere)
        # are defined. Mesh is implicitly checked elsewhere, notably.
        for title, object in [("evolver", self.evolvers),
                              ("driver", self.drivers)]:
            if not object:
                print("(%s) Warning: No %s defined." % (self.filename, title))
        
        # Check that all labels have been used
        for title, labels in [("Atlas", self.atlases),
                               ("Vector field", self.vectorFields),
                               ("Scalar field", self.scalarFields),
                               ("Evolver", map(lambda x:x[0], self.evolvers))]:
            for label in labels:
                if not label in self.labelsUsed:
                    print("(%s) Warning: %s '%s' declared, but never used." % (self.filename, title, label))
                    
        # Check that referenced files exist
        for filename in self.referencedFilenames:
            
            if filename[0] == "/" or (os.name == "nt" and filename[1] == ":"):
                isValid = os.path.isfile(filename)
            else:
                isValid = os.path.isfile(self.OOMMFStylePath + 
                                         os.path.sep + filename)
            if not isValid:
                print("(%s) Warning: Uses file %s which could not be located. Be sure to move it into position." % (self.filename, filename))
        
        # Are we outputting?
        if not self.destinations or not self.scheduled:
            print("(%s) Warning: Simulation has not configured any output" % (self.filename))
            
    
    ############################################################
    # MIF Writing Functions - Implements Oxs Extension Classes #
    ############################################################
        
    # Implemented in order of appearance in Userguide 1.2a4
    # kwargs are uniformly matched to the argument names in the userguide
       
    def purgeFile(self):
        self.hasFailed = True
        try:
            os.remove(self.filename)
        except WindowsError:
            print("Programmer error: File handle was in use while trying to clean up file in error handling.")
            print("Please send the offending MIFCraft script to doublemark@mit.edu ")
            # Don't really want to display traceback here if not debugging 
            # this problem specifically, so it doesn't re-raise.

    # Important User Notes:
    # - These functions are to be called only with kwargs, to allow internal
    #   catching of bad-name or unsupplied-argument exceptions.
    # - The "name" argument is filled in by the MIFWrite decorator.
    #   Overrides are done by passing the kwarg "name"
    # - All names given in the OOMMF specification are valid as keyword 
    #   arguments. Where additional clarity might be helpful, some names have 
    #   been given synonyms. Use of these is never required
   
    # This is very boilerplate-y, since we have to wrap every command in the 
    # OOMMF specification. Exploitable symmetries are very limited, but are
    # used when reasonable. All functions follow the same four-step flow,
    # though steps 1-3 are omitted in some cases:
    
    # 1. Pull values from keyword arguments. Generate pretty errors when the
    #    user screws this up. Raising a KeyError will give a pretty error.
    # 2. Validate all arguments: check that ranges are sane, that labels
    #    have been defined elsewhere, etc. Raising a MIFException will
    #    give a suitable pretty error.
    # -- AFTER THIS POINT, YOU MAY NOT RAISE A MIFException FOR ANY REASON --
    # -- You really shouldn't raise anything; there are no more surprises. --
    # 3. Register any new labels with the MIF object. Flag any labels used
    #    for the first time as used.
    # 4. Return a list of strings to be written to the file.
    
    # Enjoy your boilerplate. Warning: Some particularly intense
    # string substitutions exceed 80 char lines. Sorry.
    
    # 
    ###
    ##### ATLASES
    ###
    #
    
    @MIFWrite
    def BoxAtlas(self, name, **kwargs):
        """
        Implements: Oxs_BoxAtlas
         Mandatory: xrange <(min, max)>, yrange <(min, max)>, zrange <(min, max)>
          Optional: None
        """ 
        xmin, xmax = kwargs["xrange"]
        ymin, ymax = kwargs["yrange"]
        zmin, zmax = kwargs["zrange"]
        quickValidateExtent("BoxAtlas", name, xmin, xmax, 
                            ymin, ymax, zmin, zmax)
        
        self.atlases.add(name)
        self.atlasCoords[name] = (xmin, xmax, ymin, ymax, zmin, zmax)
        
        maxSignificance = sigfigs([xmin, xmax, ymin, ymax, zmin, zmax])
        
        return ["Specify Oxs_BoxAtlas:%s {" % name,
                "    xrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(xmin, xmax, sigfigs=maxSignificance),
                "    yrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(ymin, ymax, sigfigs=maxSignificance),
                "    zrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(zmin, zmax, sigfigs=maxSignificance),
                "}"]
        
    @MIFWrite
    def ImageAtlas(self, name, **kwargs):
        """
        Implements: Oxs_ImageAtlas
         Mandatory: xrange <(min, max)>, yrange <(min, max)>, zrange <(min, max)>
                    viewplane <str>, image <filename> 
                    colormap <[(color, name), (color, name) ... ]>
          Optional: matcherror <float=3>
        """        
        xmin, xmax = kwargs["xrange"]
        ymin, ymax = kwargs["yrange"]
        zmin, zmax = kwargs["zrange"]
        quickValidateExtent("ImageAtlas", name, 
                            xmin, xmax, ymin, ymax, zmin, zmax)
        
        viewplane = kwargs["viewplane"]
        if not viewplane in ["xy", "zx", "yz"]:
            raise MIFException("viewplane", 
                               "viewplane %s is not one of xy, zx, yx" % viewplane)
        
        image = kwargs["image"].replace("\\", "/") # Take care - OOMMF uses /
        colormap = kwargs["colormap"]
        
        matcherror = kwargs.get("matcherror", 3)
        
        # Possibility for errors is past - add to names
        self.atlases.add(name)
        self.atlasCoords[name] = (xmin, xmax, ymin, ymax, zmin, zmax)
        
        maxSignificance = sigfigs([xmin, xmax, ymin, ymax, zmin, zmax])
        workRet = ["Specify Oxs_ImageAtlas:%s {" % name,
                   "    xrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(xmin, xmax, sigfigs=maxSignificance),
                   "    yrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(ymin, ymax, sigfigs=maxSignificance),
                   "    zrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(zmin, zmax, sigfigs=maxSignificance),
                   "    viewplane %s" % viewplane,
                   "    image %s" % image,
                   "    colormap {"]
        
        maxWidth = max(map(lambda x: len(x[0]), colormap))
        for color, regionName in colormap:
            workRet.append("        {0!s:<{fillwidth}} {1!s}".format(color, 
                                                                     regionName, 
                                                                     fillwidth=maxWidth))
            self.regions[name].add(regionName)
        
        workRet.extend(["    }", "    matcherror %s" % matcherror, "}"])
        return workRet
    
    @MIFWrite
    def MultiAtlas(self, name, **kwargs):
        """
        Implements: Oxs_MultiAtlas
         Mandatory: atlases <[name, name, name...]>
          Optional: xrange <(min, max)>, yrange <(min, max)>, zrange <(min, max)>*
          * See user manual for explanation of optional arguments
        """
        # Here's a new one - constructing an atlas out of other atlases
        atlases = kwargs["atlases"]
        # First do validity check on input atlases
        for atlas in atlases:
            if not atlas in self.atlases:
                raise UndefTokenMIFException("atlas", "atlas", list(self.atlases))
            else:
                # Anything used as part of a multiatlas is now used
                self.labelsUsed.add(atlas)
        
        # We're safe, and can add ourselves to the set
        self.atlases.add(name)
        for atlas in atlases:
            self.regions[name].add(atlas)
            # All constituent atlases are now regions of us
            for subregion in self.regions[atlas]:
                self.regions[name].add(subregion)
            
        workRet = ["Specify Oxs_MultiAtlas:%s {" % name]
        
        for atlas in atlases:
            workRet.append("    atlas %s" % atlas)
        
        if "xrange" in kwargs:
            xmin, xmax = kwargs["xrange"]
            workRet.append("    xrange {{ {0:< .4e} {1:< .4e} }}".format(xmin, 
                                                                         xmax))
        else:
            xmin = min(map(lambda atlas: self.atlasCoords[atlas][0], atlases))
            xmax = max(map(lambda atlas: self.atlasCoords[atlas][1], atlases))
            
        if "yrange" in kwargs:
            ymin, ymax = kwargs["yrange"]
            workRet.append("    yrange {{ {0:< .4e} {1:< .4e} }}".format(ymin, 
                                                                         ymax))
        else:
            ymin = min(map(lambda atlas: self.atlasCoords[atlas][2], atlases))
            ymax = max(map(lambda atlas: self.atlasCoords[atlas][3], atlases))
            
        if "zrange" in kwargs:
            zmin, zmax = kwargs["zrange"]
            workRet.append("    zrange {{ {0:< .4e} {1:< .4e} }}".format(zmin, 
                                                                         zmax))
        else:
            zmin = min(map(lambda atlas: self.atlasCoords[atlas][4], atlases))
            zmax = max(map(lambda atlas: self.atlasCoords[atlas][5], atlases))
        quickValidateExtent("MultiAtlas", name, 
                            xmin, xmax, ymin, ymax, zmin, zmax)
            
        workRet.append("}")
        
        # Magic done, we can now do this:
        self.atlasCoords[name] = (xmin, xmax, ymin, ymax, zmin, zmax)
        
        return workRet
    
    @MIFWrite
    def ScriptAtlas(self, name, **kwargs):
        """
        Implements: Oxs_ScriptAtlas
         Mandatory: xrange <(min, max)>, yrange <(min, max)>, zrange <(min, max)>,
                    regions <[string, string, ...]>, script <script>,
          Optional: script_args <["relpt" and/or "rawpt" a/o "minpt" a/o "maxpt" a/o "span", ["relpt"]>
        """
        
        xmin, xmax = kwargs["xrange"]
        ymin, ymax = kwargs["yrange"]
        zmin, zmax = kwargs["zrange"]
        quickValidateExtent("BoxAtlas", name, 
                            xmin, xmax, ymin, ymax, zmin, zmax)        
     
        script = kwargs["script"]   
        if not script in self.scripts:
            raise UndefTokenMIFException("script", "script", list(self.scripts))
        script_args = kwargs.get("script_args", ["relpt"])
        for i, arg in enumerate(script_args):
            if not arg in ["relpt", "rawpt", "minpt", "maxpt", "span"]:
                raise MIFException("relpt", "argument", ["relpt", "rawpt", "minpt", "maxpt", "span"])
            if arg in script_args[:i]:
                raise MIFException("script_args", "%s appears more than once in script_args" % arg)            
        regions = kwargs["regions"]
        for region in regions:
            self.regions[name].add(region)
        
        self.atlases.add(name)
        self.atlasCoords[name] = (xmin, xmax, ymin, ymax, zmin, zmax)

        maxSignificance = sigfigs([xmin, xmax, ymin, ymax, zmin, zmax])
        return ["Specify Oxs_ScriptAtlas:%s {" % name,
                "    xrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(xmin, xmax, sigfigs=maxSignificance),
                "    yrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(ymin, ymax, sigfigs=maxSignificance),
                "    zrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(zmin, zmax, sigfigs=maxSignificance),
                "    regions { %s }" % " ".join(regions),
                "    script_args { %s }" % " ".join(script_args),
                "    script %s" % script,
                "}"]
    
    @MIFWrite
    def EllipsoidAtlas(self, name, **kwargs):
        """
        Implements: Oxs_EllipsoidAtlas 
         Mandatory: xrange <(min, max)>, yrange <(min, max)>, zrange <(min, max)>
          Optional: None
        """         

        xmin, xmax = kwargs["xrange"]
        ymin, ymax = kwargs["yrange"]
        zmin, zmax = kwargs["zrange"]
        quickValidateExtent("EllipsoidAtlas", name, 
                            xmin, xmax, ymin, ymax, zmin, zmax)
        
        self.atlases.add(name)
        
        maxSignificance = sigfigs([xmin, xmax, ymin, ymax, zmin, zmax])
        return ["Specify Oxs_EllipsoidAtlas:%s {" % name,
                "    xrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(xmin, xmax, sigfigs=maxSignificance),
                "    yrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(ymin, ymax, sigfigs=maxSignificance),
                "    zrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(zmin, zmax, sigfigs=maxSignificance),
                "}"]
        
    # 
    ###
    ##### MESHES
    ###
    #        
    
    @MIFWrite
    def RectangularMesh(self, name, **kwargs):
        """
        Implements: Oxs_RectangularMesh
         Mandatory: cellsize <(xstep, ystep, zstep)>, atlas <name>
          Optional: None
        """         
        
        xstep, ystep, zstep = kwargs["cellsize"]
        atlas = kwargs["atlas"]
        if not atlas in self.atlases:
            raise UndefTokenMIFException("atlas", "atlas", list(self.atlases))
        self.labelsUsed.add(atlas)
        
        if atlas == "universe":
            xmin = min(map(lambda at: self.atlasCoords[at][0], self.atlases))
            xmax = max(map(lambda at: self.atlasCoords[at][1], self.atlases))
            ymin = min(map(lambda at: self.atlasCoords[at][2], self.atlases))
            ymax = max(map(lambda at: self.atlasCoords[at][3], self.atlases))
            zmin = min(map(lambda at: self.atlasCoords[at][4], self.atlases))
            zmax = max(map(lambda at: self.atlasCoords[at][5], self.atlases))            
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = self.atlasCoords[atlas]
            
        xdiff, ydiff, zdiff = (xmax-xmin, ymax-ymin, zmax-zmin)
        
        if carefulFloatMod(xdiff, xstep):
            raise MIFException("xstep", "%s does not evenly divide atlas x size %s" % (xstep, xdiff))
        if carefulFloatMod(ydiff, ystep):
            raise MIFException("ystep", "%s does not evenly divide atlas y size %s" % (ystep, ydiff))
        if carefulFloatMod(zdiff, zstep):
            raise MIFException("zstep", "%s does not evenly divide atlas z size %s" % (zstep, zdiff))                
        
        if self.meshes:
            print("Warning: Defining mesh %s, but have defined other meshes:" % (name, list(self.meshes)))
        self.meshes.append(name)
        
        return ["Specify Oxs_RectangularMesh:%s {" % name,
                "    cellsize { %s %s %s }" % (xstep, ystep, zstep),
                "    atlas %s" % atlas,
                "}"]

    # 
    ###
    ##### ENERGIES
    ###
    #  
        
    @MIFWrite
    def UniaxialAnisotropy(self, name, **kwargs):
        """
        Implements: Oxs_UniaxialAnisotropy 
         Mandatory: K1 <scalarfield or number>, axis <vectorfield or (x, y, z)>
          Optional: None
        """ 
        
        k = kwargs["k"]
        axis = kwargs["axis"]
        
        if isinstance(k, basestring):
            if not k in self.scalarFields:
                raise UndefTokenMIFException("k", "scalar field", 
                                             list(self.scalarFields))
            else:
                self.labelsUsed.add(k)
        if isinstance(axis, basestring):                
            if not axis in self.vectorFields:
                raise UndefTokenMIFException("axis", "vector field", 
                                             list(self.vectorFields))
            else:
                self.labelsUsed.add(axis)
        else:
            axis = "{%s %s %s}" % (axis[0], axis[1], axis[2])
        
        return ["Specify Oxs_UniaxialAnisotropy:%s {" % name,
                "    K1 %s" % k,
                "    axis %s" % axis,
                "}"]
    
    @MIFWrite
    def CubicAnisotropy(self, name, **kwargs):
        """
        Implements: Oxs_CubicAnisotropy 
         Mandatory: K1 <scalarfield>, axis1 <vectorfield>, axis2 <vectorfield>
          Optional: None
        """ 
        
        k = kwargs["k"]
        axis1 = kwargs["axis1"]
        axis1 = kwargs["axis2"]
        
        if not k in self.scalarFields:
            raise UndefTokenMIFException("k", "scalar field", 
                                         list(self.scalarFields))
        else:
            self.labelsUsed.add(k)
        # Now slightly regretting exception structure; makes this
        # hard to zip in a tuple.
        if not axis1 in self.vectorFields:
            raise UndefTokenMIFException("axis1", "vector field", 
                                         list(self.vectorFields))
        else:
            self.labelsUsed.add(axis1)
            
        if not axis2 in self.vectorFields:
            raise UndefTokenMIFException("axis2", "vector field", 
                                         list(self.vectorFields))
        else:
            self.labelsUsed.add(axis2)
       
        return ["Specify CubicAnisotropy:%s {" % name,
                "    K1 %s" % k,
                "    axis1 %s" % axis1,
                "    axis2 %s" % axis2,
                "}"]
        
    @MIFWrite
    def Exchange6Ngbr(self, name, **kwargs):
        """
        Implements: Oxs_Exchange6Ngbr
         Mandatory: default_A <float> OR default_lex <float>, atlas <atlas>
                    A <[(region, region, value), (region, region, value) ...]> OR lex <...>
          Optional: 'default' may be used instead of 'default_A' or 'default_lex' for simplicity
        """
        
        atlas = kwargs["atlas"]
        if "A" in kwargs:
            typestring = "A"
            regions = kwargs["A"]
            if "default_A" in kwargs:
                default = kwargs["default_A"]
            else:
                default = kwargs["default"]
        elif "lex" in kwargs:
            typestring = "lex"
            regions = kwargs["lex"]
            if "default_lex" in kwargs:
                default = kwargs["default_lex"]
            else:
                default = kwargs["default"]
        else:
            raise NameError("Neither A nor lex provided")
        
        if not atlas in self.atlases:
            raise UndefTokenMIFException("atlas", "atlas", list(self.atlases))
        else:
            self.labelsUsed.add(atlas)
        
        # Checking region validity is a little weirder
        for A, B, val in regions:
            if not A in self.regions[atlas]:
                raise UndefTokenMIFException("A", "region name", 
                                             list(self.regions[atlas]))
            if not B in self.regions[atlas]:
                raise UndefTokenMIFException("B", "region name", 
                                             list(self.regions[atlas]))
            
        
        retWork = ["Specify Oxs_Exchange6Ngbr:%s {" % name,
                   "    default_%s %s" % (typestring, default),
                   "    atlas %s" % atlas,
                   "    %s {" % typestring]
        
        maxWidth = max(map(lambda x: max(len(x[0]),len(x[1])), regions))
        
        for A, B, val in regions:
            retWork.append("        {0!s:<{fillwidth}} {1!s:<{fillwidth}} {2!s}".format(A, B, val, fillwidth=maxWidth))
        
        retWork.extend(["    }", "}"])
        return retWork
    
    @MIFWrite
    def UniformExchange(self, name, **kwargs):
        """
        Implements: Oxs_UniformExchange
         Mandatory: A <float> or lex <float>
          Optional: None
        """ 
        
        if "A" in kwargs:
            val = kwargs["A"]
            typestring = "A"
        elif "lex" in kwargs:
            val = kwargs["lex"]
            typestring = "lex"
            
        return ["Specify Oxs_UniformExchange:%s {" % name,
                "    %s %s" % (typestring, val),
                "}"]
        
    @MIFWrite
    def ExchangePtwise(self, name, **kwargs):
        """
        Implements: Oxs_ExchangePtwise
         Mandatory: A <scalarfield>
          Optional: None
        """ 
        
        A = kwargs["A"]
        
        if not A in self.scalarFields:
            raise UndefTokenMIFException("A", "scalar field", 
                                         list(self.scalarFields))
        else:
            self.labelsUsed.add(A)
        
        return ["Specify Oxs_ExchangePtwise:%s {" % name,
                "    A %s" % A,
                "}"]
        
    # TODO: Oxs_TwoSurfaceExchange not implemented
    
    @MIFWrite
    def RandomSiteExchange(self, name, **kwargs):
        """
        Implements: Oxs_RandomSiteExchange
         Mandatory: linkprob <0 to 1>, Amin <float>, Amax <float>
          Optional: None
        """ 
        
        linkprob = kwargs["linkprob"]
        
        if not 0.0 <= linkprob <= 1.0:
            raise MIFException("linkprob", 
                               "%s is not between 0 and 1" % linkprob)
        
        Amin = kwargs["Amin"]
        Amax = kwargs["Amax"]
        
        return ["Specify Oxs_RandomSiteExchange:%s {" % name,
                "    linkprob %s" % linkprob,
                "    Amin %s" % Amin,
                "    Amax %s" % Amax,
                "}"]
        
    @MIFWrite
    def Demag(self, name, **kwargs):
        """
        Implements: Oxs_Demag 
         Mandatory: None 
          Optional: None
        """
        
        return ["Specify Oxs_Demag:%s {}" % name]
    
    @MIFWrite
    def SimpleDemag(self, name, **kwargs):
        """
        Implements: Oxs_SimpleDemag 
         Mandatory: None 
          Optional: None
        """
        
        return ["Specify Oxs_SimpleDemag:%s {}" % name]
    
    @MIFWrite
    def UZeeman(self, name, **kwargs):
        """
        Implements: Oxs_UZeeman
         Mandatory: Hrange <[(x, y, z, X, Y, Z, steps), (x, y, z, X, Y, Z, steps) ... ]>
          Optional: multiplier <float>
        """
        
        multiplier = kwargs.get("multiplier", 1)
        Hrange = kwargs["Hrange"]

        # Similar trick to sigfigs - determine minimum width needed
        # to print numbers to given significance
        maxWidth = max(map(lambda x: max(map(lambda y: len(str(y)), x[:-1])), 
                           Hrange))
        
        retWork = ["Specify Oxs_UZeeman:%s {" % name,
                   "    multiplier %s" % multiplier,
                   "    Hrange {"]
        
        stages = 0
        for brick in Hrange:
            # This isn't 80 character so hard it isn't even 160 characters
            retWork.append("        {{ {0[0]:< .{fw}e} {0[1]:< .{fw}e} {0[2]:< .{fw}e} {0[3]:< .{fw}e} {0[4]:< .{fw}e} {0[5]:< .{fw}e} {0[6]!s:<} }}".format(brick, fw=maxWidth))
            stages += brick[6]
            
        self.stageCounts.add(stages)
        
        retWork.extend(["    }", "}"])
        return retWork
    
    @MIFWrite
    def FixedZeeman(self, name, **kwargs):
        """
        Implements: Oxs_FixedZeeman
         Mandatory: field <field>
          Optional: multiplier <float>
        """
        
        field = kwargs["field"]
        multiplier = kwargs.get("multiplier", 1)
        
        if not field in self.vectorFields:
            raise UndefTokenMIFException("field", "vector field", 
                                         list(self.vectorFields))
        self.labelsUsed.add(field)
        
        return ["Specify Oxs_FixedZeeman:%s {" % name,
                "    field %s" % field,
                "    multiplier %s" % multiplier,
                "}"]
        
    @MIFWrite
    def ScriptUZeeman(self, name, **kwargs):
        """
        Implements: Oxs_ScriptUZeeman 
         Mandatory: script <script>
          Optional: multiplier <float=1>, stage_count <int>
                    script_args <["stage" and/or "total_time" and/or "stage_time"]>
        """
        
        script = kwargs["script"]
        script_args = kwargs.get("script_args", 
                                 ["stage", "stage_time", "total_time"])
        
        for i,v in enumerate(script_args):
            if not v in set(["stage", "stage_time", "total_time"]):
                raise MIFException("script_args", 
                                   "%s is not 'stage', 'stage_time', or 'total_time'" % v)
            if v in script_args[:i]:
                raise MIFException("script_args", 
                                   "%s appears more than once in script_args" % v)
        
        multiplier = kwargs.get("multiplier", 1)
        stage_count = kwargs.get("stage_count")
        
        retWork =["Specify Oxs_ScriptUZeeman:%s {" % name,
                  "    script_args { %s }" % " ".join(script_args),
                  "    script %s" % script,
                  "    multiplier %s" % multiplier]
        if stage_count:
            retWork.append("    stage_count %s" % stage_count)
            if stage_count > 0:
                self.stageCounts.add(stage_count)
            
        retWork.append("}")
        return retWork
    
    @MIFWrite
    def TransformZeeman(self, name, **kwargs):
        """
        Implements: Oxs_TransformZeeman 
         Mandatory: script <script>, field <vectorfield>, type <transformtype>
          Optional: multiplier <float=1>, stage_count <int>
                    script_args <["stage" and/or "total_time" and/or "stage_time"]>
        """ 
        
        script = kwargs["script"]
        script_args = kwargs.get("script_args", 
                                 ["stage", "stage_time", "total_time"])
        
        for i,v in enumerate(script_args):
            if not v in set(["stage", "stage_time", "total_time"]):
                raise MIFException("script_args", 
                                   "%s is not 'stage', 'stage_time', or 'total_time'" % v)
            if v in script_args[:i]:
                raise MIFException("script_args", 
                                   "%s appears more than once in script_args" % v)
        
        field = kwargs["field"]
        if not field in self.vectorFields:
            raise UndefTokenMIFException("field", "vector field", 
                                         list(self.vectorFields))
        else:
            self.labelsUsed.add(field)
        
        type = kwargs["type"]
        if not type in ["identity", "diagonal", "symmetric", "general"]:
            raise MIFException("type", "%s is not one of %s" % (field, ["identity", "diagonal", "symmetric", "general"]))
        
        multiplier = kwargs.get("multiplier", 1)
        stage_count = kwargs.get("stage_count")
        
        retWork = ["Specify Oxs_TransformZeeman:%s {" % name,
                   "    field %s" % field,
                   "    script %s" % script,
                   "    script_args { %s }" % " ".join(script_args),
                   "    multiplier %s" % multiplier]
        
        if stage_count:
            retWork.append("    stage_count %s" % stage_count)
            if stage_count > 0:
                self.stageCounts.add(stage_count)
                        
        retWork.append("}")
        return retWork
    
    @MIFWrite
    def StageZeeman(self, name, **kwargs):
        """
        Implements: Oxs_StageZeeman 
         Mandatory: script <script> OR files <[filename, filename ...]>
          Optional: multiplier <float=1>, stage_count <int>
        """
        
        if "script" in kwargs and "files" in kwargs:
            raise MIFException("script AND files", 
                               "Specified both 'script' and 'files'.")
        
        multiplier = kwargs.get("multiplier", 1)
        stage_count = kwargs.get("stage_count")
        
        retWork = ["Specify Oxs_StageZeeman:%s {" % name]
        
        if "script" in kwargs:
            retWork.append("    script %s" % script)
        else:
            files = kwargs["files"]
            filetable = ["    files {"]
            for fname in files:
                filetable.append("        %s" % file)
                self.referencedFilenames.add(fname)
            filetable.append["    }"]
            retWork.extend(filetable)
        
        if "stage_count" in kwargs:
            retWork.append("    stage_count %s" % stage_count)
            if stage_count > 0:
                self.stageCounts.add(stage_count)
                
        retWork.extend(["    multiplier %s" % multiplier,
                        "}"])
        
        return retWork
    
    # 
    ###
    ##### EVOLVERS
    ###
    #
    
    @MIFWrite
    def EulerEvolve(self, name, **kwargs):
        """
        Implements: Oxs_EulerEvolve
         Mandatory: None
          Optional: gamma_LL <float=2.211E5> OR gamma_G <float>, alpha <float=0.5>, do_precess <bool=1>
                    min_timestep <float> AND max_timestep <float>
                    fixed_spins <(atlas, [region, region ... ])>
                    start_dm <float=0.01>
                    error_rate <float=1>, absolute_step_error <float=0.2>, relative_step_error <float=0.2>
                    step_headroom <0 to 1 exclusive=0.85>
        """
        
        # Due to complexity, adjusting normal argument-gathering
        # order and building piecewise
        
        retWork = ["Specify Oxs_EulerEvolve:%s {" % name]
        
        if "gamma_LL" in kwargs and "gamma_G" in kwargs:
            raise MIFException("gamma_LL AND gamma_G", 
                               "Specified both 'gamma_LL' and ''.")
        
        retWork.append("    alpha %s" % kwargs.get("alpha", 0.5))
        
        if "gamma_G" in kwargs:
            retWork.append("    gamma_G %s" % kwargs["G"])
        elif "gamma_LL" in kwargs:
            retWork.append("    gamma_LL %s" % kwargs["gamma_LL"])
        
        do_precess = kwargs.get("do_precess", 1)
        if not (do_precess == 0 or do_precess == 1):
            raise MIFException("do_precess", "%s is not 0 or 1" % do_precess)
        retWork.append("    do_precess %s" % do_precess)
        
        min_timestep = kwargs.get("min_timestep")
        max_timestep = kwargs.get("max_timestep")
        
        if min_timestep or max_timestep:
            if (min_timestep == None) ^ (max_timestep == None):
                raise MIFException("min_timestep AND max_timestep", 
                                   "Specified one but not both of min_timestep, max_timestep")
            retWork.append("    min_timestep %s" % min_timestep)
            retWork.append("    max_timestep %s" % max_timestep)
            
        fixed_spins = kwargs.get("fixed_spins")
        if fixed_spins:
            atlas = fixed_spins[0]
            regions = fixed_spins[1]
            self.labelsUsed.add(atlas)
            
            retWork.append("    fixed_spins { %s" % atlas)
            for region in regions:
                if not region in self.regions[atlas]:
                    raise UndefTokenMIFException("region", "region", 
                                                 list(self.regions[atlas]))
                else:
                    retWork.append("        %s" % region)
            retWork.append("    }")
        
        for simpleLabel in ["start_dm", "error_rate", "absolute_step_error", 
                            "relative_step_error", "step_headroom"]:
            if simpleLabel in kwargs:
                retWork.append("    %s %s" % (simpleLabel, kwargs[simpleLabel]))
                # We can also error-check one of these
                if simpleLabel == "step_headroom" and not 0 < kwargs[simpleLabel] < 1: 
                    raise MIFException("step_headroom", 
                                       "%s is not between 0 and 1 exclusive" % kwargs[simpleLabel])
        
        retWork.append("}")
        if self.evolvers:
            print("Warning: Defining evolver %s, but have defined other evolvers: %s" % (name, map(lambda x: x[0], self.evolvers)))
        if name:
            self.evolvers.append([name, "TIME"])
        return retWork
    
    @MIFWrite
    def RungeKuttaEvolve(self, name, **kwargs):
        """
        Implements: Oxs_RungeKuttaEvolve
         Mandatory: None
          Optional: gamma_LL <float=2.211E5> OR gamma_G <float>, alpha <float=0.5>, do_precess <bool=1>
                    min_timestep <float> AND max_timestep <float>
                    fixed_spins <(atlas, [region, region ... ])>
                    start_dm <float=0.01>, method <str="rkf54">
                    error_rate <float=1>, absolute_step_error <float=0.2>, relative_step_error <float=0.01>
                    min_step_headroom <0 to 1 exclusive=0.33>, min_step_headroom <0 to 1 exclusive=0.95>
                    reject_goal <float=0.05>
        """
        
        # Due to complexity, adjusting normal argument-gathering order and building piecewise
        
        retWork = ["Specify Oxs_RungeKuttaEvolve:%s {" % name]
        
        kwargs.pop("_noName", None) # Defense against multilayer use
        kwargs.pop("_noWrite", None)
        # Finally some decent code reuse. We can partially build an EulerEvolve
        # and do work on top of it, making sure to instruct the wrapper to
        # provide neither name nor writing
        retWork.extend(self.EulerEvolve(_noWrite = True, _noName = True, 
                                        name = name, **kwargs)[1:-1])

        allow_signed_gamma = kwargs.get("allow_signed_gamma", 0)
        if not (allow_signed_gamma == 0 or allow_signed_gamma == 1):
            raise MIFException("allow_signed_gamma", 
                               "%s is not 0 or 1" % allow_signed_gamma)
        retWork.append("    allow_signed_gamma %s" % allow_signed_gamma)
        
        for simpleLabel in ["min_step_headroom", 
                            "max_step_headroom", 
                            "reject_goal"]:
            if simpleLabel in kwargs:
                retWork.append("    %s %s" % (simpleLabel, kwargs[simpleLabel]))
        
        method = kwargs.get("method", "rkf54")
        if not method in ["rk2", "rk4", "rkf54", "rkf54m", "rkf54s"]:
            raise MIFException("method", 
                               "%s not in %s" % (method, 
                                                 ["rk2", "rk4", "rkf54", "rkf54m", "rkf54s"]))
        retWork.append("    method %s" % method)
        
        retWork.append("}")
        if name:
            self.evolvers.append([name, "TIME"])
        return retWork

    @MIFWrite
    def SpinXferEvolve(self, name, **kwargs):
        """
        Implements: Oxs_SpinXferEvolve
         Mandatory: J <float>, mp <(x, y, z)>
          Optional: P <float=0.4>, Lambda <float=2> OR
                    P_fixed <float>, P_free <float>, Lambda_fixed <float>, Lambda_free <float>
                    J_profile <script>, J_profile_args <["stage" and/or "total_time" and/or "stage_time"]>
                    
                    eps_prime <undocumented>, energy_slack <undocumented>
                    
                    gamma_LL <float=2.211E5> OR gamma_G <float>, alpha <float=0.5>, do_precess <bool=1>
                    min_timestep <float> AND max_timestep <float>
                    fixed_spins <(atlas, [region, region ... ])>
                    start_dm <float=0.01>, method <str="rkf54">
                    error_rate <float=-1>, absolute_step_error <float=0.2>, relative_step_error <float=0.01>
                    min_step_headroom <0 to 1 exclusive=0.33>, min_step_headroom <0 to 1 exclusive=0.95>
                    reject_goal <float=0.05>
        """
        
        retWork = ["Specify Oxs_SpinXferEvolve:%s {" % name]
        
        kwargs.pop("_noName", None) # Defense against multilayer use
        kwargs.pop("_noWrite", None)
        # As before - extends previous evolver
        retWork.extend(self.RungeKuttaEvolve(_noWrite = True, 
                                             _noName = True, **kwargs)[1:-1])

        if "Lambda" in kwargs:
            retWork.append("    Lambda %s" % kwargs["Lambda"])
        else:
            if "Lambda_fixed" in kwargs:
                retWork.append("    Lambda_fixed %s" % kwargs["Lambda_fixed"])
            if "Lambda_free" in kwargs:
                retWork.append("    Lambda_free %s" % kwargs["Lambda_free"])

        if "P" in kwargs:
            retWork.append("    P %s" % kwargs["P"])
        else:
            if "P_fixed" in kwargs:
                retWork.append("    P_fixed %s" % kwargs["P_fixed"])
            if "P_free" in kwargs:
                retWork.append("    P_free %s" % kwargs["P_free"])
                
        retWork.append("    J %s" % kwargs["J"])
        retWork.append("    mp {{ {0} {1} {2} }}".format(*kwargs["mp"]))

        for simpleLabel in ["J_profile", "eps_prime", "energy_slack"]:
            if simpleLabel in kwargs:
                retWork.append("    %s %s" % (simpleLabel, kwargs[simpleLabel]))
        
        if "J_profile" in kwargs:        
            J_profile_args = kwargs.get("J_profile_args", 
                                        ["stage", "stage_time", "total_time"])
            
            for i,v in enumerate(J_profile_args):
                if not v in set(["stage", "stage_time", "total_time"]):
                    raise MIFException("J_profile_args", 
                                       "%s is not 'stage', 'stage_time', or 'total_time'" % v)
                if v in J_profile_args[:i]:
                    raise MIFException("J_profile_args", 
                                       "%s appears more than once in J_profile_args" % v)
                
            retWork.append("    J_profile_args { %s }" % " ".join(J_profile_args))

        retWork.append("}")
        if name:
            self.evolvers.append([name, "TIME"])
        return retWork
    
    @MIFWrite
    def CGEvolve(self, name, **kwargs):
        """
        Implements: Oxs_CGEvolve
         Mandatory: None
          Optional: gradient_reset_angle <float=80>, gradient_reset_count <int=50>
                    minimum_bracket_step <float=0.05>, maximum_bracket_step <float=10>
                    line_minimum_angle_precision <float=UNDOCUMENTED>, line_minimum_relwidth <float=UNDOCUMENTED>
                    energy_precision <float=1e-12>, method <str in ["Fletcher-Reeves", "Polak-Ribiere"]="Fletcher-Reeves">
                    fixed_spins <(atlas, [region, region ... ])>
        """
        
        retWork = ["Specify Oxs_CGEvolve:%s {" % name]

        # Quickly gather arbitrarily-ordered arguments
        for simpleLabel in ["gradient_reset_angle", "gradient_reset_count", 
                            "minimum_bracket_step", "maximum_bracket_step", 
                            "line_minimum_angle_precision", 
                            "line_minimum_relwidth",
                            "energy_precision"]:
            if simpleLabel in kwargs:
                retWork.append("    %s %s" % (simpleLabel, kwargs[simpleLabel]))
        
        method = kwargs.get("method", "Fletcher-Reeves")
        if not method in ["Fletcher-Reeves", "Polak-Ribiere"]:
            raise MIFException("method", 
                               "%s is not 'Fletcher-Reeves' or 'Polak-Ribiere'" % method)
        retWork.append("    method %s" % method)
 
        fixed_spins = kwargs.get("fixed_spins")
        if fixed_spins:
            atlas = fixed_spins[0]
            regions = fixed_spins[1:]
            
            retWork.append("    fixed_spins { %s" % atlas)
            for region in regions:
                if not region in self.regions[atlas]:
                    raise UndefTokenMIFException("region", 
                                                 "region", 
                                                 list(self.regions[atlas]))
                else:
                    retWork.append("        %s" % region)
            retWork.append("    }")
        
        retWork.append["}"]
        if name:
            self.evolvers.append([name, "MINIMIZING"])
        return retWork
    
    @MIFWrite
    def SpinTEvolve(self, name, **kwargs):
        """
        Implements: Anv_SpinTEvolve
         Mandatory: beta <float>, u <float or scalar field>
          Optional: u_profile <script>, u_profile_args <["stage" a/o "total_time" a/o "stage_time"]>
                    gamma_LL <float=2.211E5> OR gamma_G <float>, alpha <float=0.5>, do_precess <bool=1>
                    min_timestep <float> AND max_timestep <float>
                    fixed_spins <(atlas, [region, region ... ])>
                    start_dm <float=0.01>, method <str="rkf54">
                    error_rate <float=1>, absolute_step_error <float=0.2>, relative_step_error <float=0.01>
                    min_step_headroom <0 to 1 exclusive=0.33>, min_step_headroom <0 to 1 exclusive=0.95>
                    reject_goal <float=0.05>
        """
        
        retWork = ["Specify Anv_SpinTEvolve:%s {" % name]
        
        kwargs.pop("_noName", None) # Defense against multilayer use
        kwargs.pop("_noWrite", None)
        # As before, extend a previous evolver.
        retWork.extend(self.RungeKuttaEvolve(_noWrite = True, _noName = True, 
                                             **kwargs)[1:-1])
        
        u = kwargs["u"]
        if isinstance(u, basestring):
            if not u in self.scalarFields:
                raise UndefTokenMIFException("u", "scalar field", 
                                             list(self.scalarFields))
            self.labelsUsed.add(u)
        
        retWork.append("    u %s" % kwargs["u"])
        retWork.append("    beta %s" % kwargs["beta"])

        u_profile = kwargs.get("u_profile")
        if u_profile:
            if not u_profile in self.scripts:
                raise MIFException("u_profile", "script", list(self.scripts))
            retWork.append("    u_profile %s" % u_profile)


        for simpleLabel in ["eps_prime", "energy_slack"]:
            if simpleLabel in kwargs:
                retWork.append("    %s %s" % (simpleLabel, kwargs[simpleLabel]))
                
        if "u_profile" in kwargs:
            u_profile_args = kwargs.get("u_profile_args", 
                                        ["stage", "stage_time", "total_time"])
            
            for i,v in enumerate(u_profile_args):
                if not v in set(["stage", "stage_time", "total_time"]):
                    raise MIFException("u_profile_args", 
                                       "%s is not 'stage', 'stage_time', or 'total_time'" % v)
                if v in u_profile_args[:i]:
                    raise MIFException("u_profile_args", 
                                       "%s appears more than once in u_profile_args" % v)
                
            retWork.append("    u_profile_args { %s }" % " ".join(u_profile_args))     
        retWork.append("}")
        self.evolvers.append([name, "TIME"])
        return retWork

    # 
    ###
    ##### DRIVERS
    ###
    #
    
    
    @MIFWrite
    def TimeDriver(self, name, **kwargs):
        """
        Implements: Oxs_TimeDriver 
         Mandatory: Ms <scalar field>, m0 <vector field>
          Optional: evolver <evolver=first defined>, mesh <mesh=first defined>
                    stage_count <float=0>, stage_count_check <0 or 1=0>
                    stopping_dm_dt <float or 0=0>, stopping_time <float or 0=0>, stage_iteration_limit <int or 0=0>
                    checkpoint_file <path=OOMMF default>, checkpoint_interval <minutes, -1, or 0=15>
                    checkpoint_cleanup <"normal", "done_only", or "never"="normal">
                    normalize_aveM_output <0 or 1=1>, scalar_output_format <printf string='%.17g'>
                    vector_field_output_format <'text', 'binary 4', or 'binary 8'='binary 8'>
                    report_max_spin_angle <0 or 1=1>
        """
        
        retWork = ["Specify Oxs_TimeDriver:%s {" % name]
        
        # Hate code duplication, but not willing to go as far as 
        # using attr tricks here.
        
        if "evolver" in kwargs:
            evolver = kwargs["evolver"]
            if not evolver in map(lambda x: x[0], self.evolvers):
                raise UndefTokenMIFException("evolver", "evolver", 
                                             list(self.evolvers))
            if [evolver, "TIME"] not in self.evolvers:
                raise MIFException("evolver", 
                                   "%s isn't a time evolver" % evolver)
        else:
            if not self.evolvers:
                raise MIFException("evolver", "No evolvers have been defined")
            evolver = self.evolvers[0]
        self.labelsUsed.add(evolver[0])
        
        if "mesh" in kwargs:
            mesh = kwargs["mesh"]
            if not mesh in self.meshes:
                raise UndefTokenMIFException("mesh", "mesh", list(self.meshes))
        else:
            if not self.meshes:
                raise MIFException("mesh", "No meshes have been defined")
            mesh = self.meshes[0]  
        self.labelsUsed.add(mesh)      
        
        retWork.extend(["    evolver %s" % evolver[0], "    mesh %s" % mesh])
        
        Ms = kwargs["Ms"]
        if not Ms in self.scalarFields:
            raise UndefTokenMIFException("Ms", "scalar field", 
                                         list(self.scalarFields))
        else:
            self.labelsUsed.add(Ms)
        m0 = kwargs["m0"]
        if not m0 in self.vectorFields:
            raise UndefTokenMIFException("m0", "vector field", 
                                         list(self.vectorFields))
        else:
            self.labelsUsed.add(m0)
        
        retWork.extend(["    Ms %s" % Ms, "    m0 %s" % m0])
        
        retWork.append("    basename %s" % self.basename)
        
        # And now it gets ugly. Read a bunch of things, compare to a bunch
        # of very specific other things, possibly raise exceptions. 
        
        # TODO: There are cool ways to do this with tuples, although they don't
        # save *much* mess. 
        if "checkpoint_file" in kwargs:
            retWork.append("    checkpoint_file %s" % kwargs["checkpoint_file"])
        if "checkpoint_interval" in kwargs:
            ci = kwargs["checkpoint_interval"]
            if not (ci == -1 or ci >= 0):
                raise MIFException("checkpoint_interval", 
                                   "%s is not -1 or >= 0" % ci)
            retWork.append("    checkpoint_interval %s" % ci)
        if "checkpoint_cleanup" in kwargs:
            cc = kwargs["checkpoint_cleanup"]
            if not cc in ["normal", "done_only", "never"]:
                raise MIFException("checkpoint_cleanup", 
                                   "%s not one of %s" % (cc, ["normal", "done_only", "never"]))
            retWork.append("    checkpoint_cleanup %s" % cc)
        if "normalize_aveM_output" in kwargs:
            nao = kwargs["normalize_aveM_output"]
            if not (nao == 0 or nao == 1):
                raise MIFException("normalize_aveM_output", 
                                   "%s is not 0 or 1" % nao)
            retWork.append("    normalize_aveM_output %s" % nao)
        if "scalar_output_format" in kwargs:
            retWork.append("    scalar_output_format %s" % kwargs["scalar_output_format"])
        if "vector_field_output_format" in kwargs:
            vfof = kwargs["vector_field_output_format"]
            if 0 and not vfof in ['text', 'binary 4', 'binary 8']: # line hacked Sumit 3/29/2017
                raise MIFException("vector_field_output_format", 
                                   "%s is not one of %s" % (vfof, ['text', 'binary 4', 'binary 8']))
            retWork.append("    vector_field_output_format { %s }" % vfof)
        if "report_max_spin_angle" in kwargs:
            rmsa = kwargs["report_max_spin_angle"]
            if not (rmsa == 0 or rmsa == 1):
                raise MIFException("report_max_spin_angle", 
                                   "%s is not 0 or 1" % rmsa)
            retWork.append("    report_max_spin_angle %s" % rmsa)
        if "total_iteration_limit" in kwargs:
            til = kwargs["total_iteration_limit"]
            if not (til >= 0):
                raise MIFException("total_iteration_limit", 
                                   "%s not >= 0" % til)
            retWork.append("    total_iteration_limit %s" % til)
        
        # Finally complicated bits. We need to do the right thing with 
        # stage_count and stage_count_check. 
        if kwargs.get("stage_count_check"):
            retWork.append("    stage_count_check 1")
            # ...but we're going to check it too
            sc = kwargs["stage_count"]
            if not sc > 0:
                    raise MIFException("stage_count", "%s not >= 0" % sc)
            if sc < max(self.stageCounts):
                raise MIFException("stage_count", "Gave stage count %s, but other specifications at least %s" % (sc, max(self.stageCounts)))
            elif sc > max(self.stageCounts):
                print("WARNING: Gave stage count %s, but all other specifications require at most %s" % (sc, min(self.stageCounts)))
            retWork.append("    stage_count %s" % stage_count)
        else:
            if "stage_count" in kwargs:
                stage_count = kwargs["stage_count"]
                if not stage_count > 0:
                    raise MIFException("stage_count", "%s not >= 0" % stage_count)
                retWork.append("    stage_count %s" % stage_count)
        
        # We also need to do the right thing with stopping criteria
        if "stopping_dm_dt" not in kwargs and "stopping_time" not in kwargs \
          and "stage_iteration_limit" not in kwargs:
            print("Warning: No stopping criteria specified. This is not technically an error.")
        
        for var in ["stopping_dm_dt", "stopping_time", "stage_iteration_limit"]:
            if var in kwargs:
                val = kwargs[var]
                if hasattr(val, "__iter__"):
                    #Using listed-rules type
                    if any(map(lambda x: True if x < 0 else False, val)):
                        raise MIFException(var, "%s has values not >= 0" % val)
                    retWork.append("    %s { %s }" % (var, " ".join(val)))
                else:
                    #Using single value type
                    if val < 0:
                        raise MIFException(var, "%s not >= 0" % val)
                    retWork.append("    %s %s" % (var, val))
        
        retWork.append("}")
        self.drivers.append(name)
        return retWork
        
    @MIFWrite
    def MinDriver(self, name, **kwargs):
        """
        Implements: Oxs_MinDriver 
         Mandatory: Ms <scalar field>, m0 <vector field>
          Optional: evolver <evolver=first defined>, mesh <mesh=first defined>
                    stage_count <float=0>, stage_count_check <0 or 1=0>
                    stopping_mxHxm <float=0>, stage_iteration_limit <int or 0=0>
                    checkpoint_file <path=OOMMF default>, checkpoint_interval <minutes, -1, or 0=15>
                    checkpoint_cleanup <"normal", "done_only", or "never"="normal">
                    normalize_aveM_output <0 or 1=1>, scalar_output_format <printf string='%.17g'>
                    vector_field_output_format <'text', 'binary 4', or 'binary 8'='binary 8'>
                    report_max_spin_angle <0 or 1=1>
        """
        
        retWork = ["Specify Oxs_MinDriver:%s {" % name]
        
        # Hate code duplication, but not willing to go as far as using attr tricks here
        
        if "evolver" in kwargs:
            evolver = kwargs["evolver"]
            if not evolver in self.evolvers:
                raise UndefTokenMIFException("evolver", "evolver", 
                                             list(self.evolvers))
            if [evolver, "MINIMIZING"] not in self.evolvers:
                raise MIFException("evolver", 
                                   "%s isn't a minimizing evolver" % evolver)            
        else:
            if not self.evolvers:
                raise MIFException("evolver", "No evolvers have been defined")
            evolver = self.evolvers[0]
        
        if "mesh" in kwargs:
            mesh = kwargs["mesh"]
            if not mesh in self.meshes:
                raise UndefTokenMIFException("mesh", "mesh", list(self.meshes))
        else:
            if not self.meshes:
                raise MIFException("mesh", "No meshes have been defined")
            mesh = self.mesh[0]        
        
        retWork.extend(["    evolver %s" % evolver[0], "    mesh %s" % mesh])
        
        Ms = kwargs["Ms"]
        if not Ms in self.scalarFields:
            raise UndefTokenMIFException("Ms", "scalar field", 
                                         list(self.scalarFields))
        else:
            self.labelsUsed.add(Ms)
        m0 = kwargs["m0"]
        if not m0 in self.vectorFields:
            raise UndefTokenMIFException("m0", "vector field", 
                                         list(self.vectorFields))
        else:
            self.labelsUsed.add(m0)
        
        retWork.extend(["    Ms %s" % Ms, "    m0 %s" % m0])
        
        retWork.append("    basename %s" % self.basename)
        
        if "checkpoint_file" in kwargs:
            retWork.append("    checkpoint_file %s" % kwargs["checkpoint_file"])
        if "checkpoint_interval" in kwargs:
            ci = kwargs["checkpoint_interval"]
            if not (ci == -1 or ci >= 0):
                raise MIFException("checkpoint_interval", 
                                   "%s is not -1 or >= 0" % ci)
            retWork.append("    checkpoint_interval %s" % ci)
        if "checkpoint_cleanup" in kwargs:
            cc = kwargs["checkpoint_cleanup"]
            if not cc in ["normal", "done_only", "never"]:
                raise MIFException("checkpoint_cleanup", 
                                   "%s not one of %s" % (cc, 
                                                         ["normal", "done_only", "never"]))
            retWork.append("    checkpoint_cleanup %s" % cc)
        if "normalize_aveM_output" in kwargs:
            nao = kwargs["normalize_aveM_output"]
            if not (nao == 0 or nao == 1):
                raise MIFException("normalize_aveM_output", 
                                   "%s is not 0 or 1" % nao)
            retWork.append("    normalize_aveM_output %s" % nao)
        if "scalar_output_format" in kwargs:
            retWork.append("    scalar_output_format %s" % kwargs["scalar_output_format"])
        if "vector_field_output_format" in kwargs:
            vfof = kwargs["vector_field_output_format"]
            if not vfof in ['text', 'binary 4', 'binary 8']:
                raise MIFException("vector_field_output_format", 
                                   "%s is not one of %s" % (vfof, ['text', 'binary 4', 'binary 8']))
            retWork.append("    vector_field_output_format { %s }" % vfof)
        if "report_max_spin_angle" in kwargs:
            rmsa = kwargs["report_max_spin_angle"]
            if not (rmsa == 0 or rmsa == 1):
                raise MIFException("report_max_spin_angle", 
                                   "%s is not 0 or 1" % rmsa)
            retWork.append("    report_max_spin_angle %s" % rmsa)
        if "total_iteration_limit" in kwargs:
            til = kwargs["total_iteration_limit"]
            if not (til >= 0):
                raise MIFException("total_iteration_limit", 
                                   "%s not >= 0" % til)
            retWork.append("    total_iteration_limit %s" % til)
        
        # Finally complicated bits. We need to do the right thing with stage_count and
        # stage_count_check. 
        if kwargs.get("stage_count_check"):
            retWork.append("    stage_count_check 1")
            # ...but we're going to check it too
            sc = kwargs["stage_count"]
            if not sc > 0:
                    raise MIFException("stage_count", "%s not >= 0" % sc)
            if sc < max(self.stageCounts):
                raise MIFException("stage_count", 
                                   "Gave stage count %s, but other specifications at least %s" % (sc, max(self.stageCounts)))
            elif sc > max(self.stageCounts):
                print("WARNING: Gave stage count %s, but all other specifications require at most %s" % (sc, min(self.stageCounts)))
            retWork.append("    stage_count %s" % stage_count)
        else:
            if "stage_count" in kwargs:
                sc = kwargs["stage_count"]
                if not sc > 0:
                    raise MIFException("stage_count", "%s not >= 0" % sc)
                retWork.append("    stage_count %s" % stage_count)
        
        # We also need to do the right thing with stopping criteria
        if "stopping_mxHxm" not in kwargs and "stage_iteration_limit" not in kwargs:
            print("Warning: No stopping criteria specified. This is not technically an error.")
        
        for var in ["stopping_mxHxm", "stage_iteration_limit"]:
            if var in kwargs:
                val = kwargs[var]
                if val < 0:
                    raise MIFException(var, "%s not >= 0" % val)
                retWork.append("    %s %s" % (var, val))
        
        retWork.append("}")
        self.drivers.append(name)
        return retWork
    
    # 
    ###
    ##### SCALAR FIELDS
    ###
    #
    
    @MIFWrite
    def UniformScalarField(self, name, **kwargs):
        """
        Implements: Oxs_UniformScalarField
         Mandatory: value <float>
          Optional: None
        """        
        
        self.scalarFields.add(name)
        value = kwargs["value"]
        
        return ["Specify Oxs_UniformScalarField:%s {" % name,
                "    value %s" % value,
                "}"]
 
    @MIFWrite
    def AtlasScalarField(self, name, **kwargs):
        """
        Implements: Oxs_AtlasScalarField
         Mandatory: atlas <atlas>, values <[(region, value), (region, value) ... ]>
          Optional: default_value <float>, multiplier <float>
        """        
        
        atlas = kwargs["atlas"]
        pairs = kwargs["values"]
        
        if not atlas in self.atlases:
            raise UndefTokenMIFException("atlas", "atlas", list(self.atlases))
        
        for region, value in pairs:
            if not region in self.regions[atlas]:
                raise UndefTokenMIFException("region", "region name", 
                                             list(self.regions[atlas]))
        self.labelsUsed.add(atlas)
        
        self.scalarFields.add(name)
        
        retWork= ["Specify Oxs_AtlasScalarField:%s {" % name,
                  "    atlas %s" % atlas]
        if kwargs.get('default_value'):
            retWork.append("    default_value %s" % kwargs['default_value'])
        retWork.extend(["    multiplier %s" % kwargs.get("multiplier", 1),
                        "    values {"])
        
        for region, value in pairs:
            retWork.append("    %s %s" % (region, value))
        
        retWork.extend(["    }", "}"])
        return retWork   
    
    @MIFWrite
    def LinearScalarField(self, name, **kwargs):
        """
        Implements: Oxs_LinearScalarField 
         Mandatory: vector <(x, y, z)>
          Optional: norm <float>, offset <float=0>
        """        
        
        self.scalarFields.add(name)
        x, y, z = kwargs["vector"]
        retWork = ["Specify Oxs_LinearScalarField:%s {" % name,
                   "    vector { %s %s %s }" % (x, y, z)]
        
        if "norm" in kwargs:
            retWork.append("    norm %s" % norm)
            
        if "offset" in kwargs:
            retWork.append("    offset %s" % offset)
        
        retWork.append("}")
        return retWork
    
    @MIFWrite
    def RandomScalarField(self, name, **kwargs):
        """
        Implements: Oxs_RandomScalarField 
         Mandatory: range_min <float>, range_max <float>
          Optional: cache_grid <mesh>
        """
        
        range_min = kwargs["range_min"]
        range_max = kwargs["range_max"]
        
        if range_min > range_max:
            raise MIFException("range_min and range_max", 
                               "minimum %s is larger than maximum %s" % (range_min, range_max))
        
        retWork = ["Specify Oxs_RandomScalarField:%s {" % name,
                   "    range_min %s" % range_min,
                   "    range_max %s" % range_max]
        
        if "cache_grid" in kwargs:
            cache_grid = kwargs["cache_grid"]
            if not cache_grid in self.meshes:
                raise UndefTokenMIFException("cache_grid", 
                                             "mesh", list(self.meshes))
            retWork.append("    cache_grid %s" % cache_grid)
        self.scalarFields.add(name)
                    
        retWork.append("}")
        return retWork
    
    @MIFWrite
    def ScriptScalarField(self, name, **kwargs):
        """
        Implements: Oxs_ScriptScalarField
         Mandatory: script <script>
          Optional: script_args <["relpt" and/or "rawpt" a/o "minpt" a/o "maxpt" a/o "span"
                                  a/o "scalar_fields" a/o "vector_fields"], ["relpt"]>
                    atlas <atlas> OR xrange <(float, float)>, yrange <(float, float)>, zrange <(float, float)>
                    scalar_fields <[field, field, ... ]>, vector_fields <[field, field, ... ]>
                    multiplier <float=1>
        """
        script = kwargs["script"]
        if not script in self.scripts:
            raise UndefTokenMIFException("script", "script", list(self.scripts))        
        script_args = kwargs.get("script_args", ["relpt"])
        
        multiplier = kwargs.get("multiplier", 1)
        
        for i, v in enumerate(script_args):
            if not v in set(["relpt", "rawpt", "minpt", "maxpt", "span",
                              "scalar_fields", "vector_fields"]):
                raise MIFException("script_args", "%s is not 'relpt', 'rawpt', 'minpt', 'maxpt', or 'span'" % v)
            if v in script_args[:i]:
                raise MIFException("script_args", "%s appears more than once in script_args" % v)
            
        retWork = ["Specify Oxs_ScriptScalarField:%s {" % name,
                   "    script %s" % script,
                   "    multiplier %s" % multiplier]
        
        
        if not "scalar_fields" in script_args and not "vector_fields" in script_args:
            retWork.append("    script_args { %s }" % " ".join(script_args))
        else:
            retWork.append("    script_args {")
            # Can't reorder, so can't do a list comp
            for label in script_args:
                if label not in ["scalar_fields", "vector_fields"]:
                    retWork.append("        %s" % label)
                elif label == "scalar_fields":
                    if not "scalar_fields" in kwargs:
                        raise MIFException("scalar_fields", 
                                           "Used scalar_fields in script_args, but didn't supply any")
                    scalar_fields = kwargs["scalar_fields"]
                    retWork.append("        scalar_fields {")
                    for scalar_field in scalar_fields:
                        if scalar_field not in self.scalarFields:
                            raise UndefTokenMIFException("scalar_field", 
                                                         "scalar field", 
                                                         list(self.scalarFields))
                        retWork.append("            %s" % scalar_field)
                        self.labelsUsed.add(scalar_field)
                    retWork.append("        }")
                elif label == "vector_fields":
                    if not "vector_fields" in kwargs:
                        raise MIFException("vector_fields", "Used vector_fields in script_args, but didn't supply any")
                    vector_fields = kwargs["vector_fields"]
                    retWork.append("        vector_fields {")
                    for vector_field in vector_fields:
                        if vector_field not in self.scalarFields:
                            raise UndefTokenMIFException("vector_field", 
                                                         "vector field", 
                                                         list(self.scalarFields))
                        retWork.append("            %s" % vector_field)
                        self.labelsUsed.add(vector_field)
                    retWork.append("        }")                    
            retWork.append("    }")
        
        if "atlas" in kwargs:
            atlas = kwargs["atlas"]
            if not atlas in self.atlases:
                raise UndefTokenMIFException("atlas", "atlas", 
                                             list(self.atlases))
            self.labelsUsed.add(atlas)
            retWork.append("    atlas %s" % atlas)
        elif "xrange" in kwargs and "yrange" in kwargs and "zrange" in kwargs:
            xmin, xmax = kwargs["xrange"]
            ymin, ymax = kwargs["yrange"]
            zmin, zmax = kwargs["zrange"]
            quickValidateExtent("ScriptScalarField", name, xmin, xmax, 
                                ymin, ymax, zmin, zmax)
        
            maxSignificance = sigfigs([xmin, xmax, ymin, ymax, zmin, zmax])
            
            retWork.extend(["    xrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(xmin, xmax, sigfigs=maxSignificance),
                            "    yrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(ymin, ymax, sigfigs=maxSignificance),
                            "    zrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(zmin, zmax, sigfigs=maxSignificance)])
        else:
            raise MIFException("atlas or xrange/yrange/zrange", "Didn't specify atlas or all of (xrange, yrange, zrange)")
        
        self.scalarFields.add(name)
        retWork.append("}")       
    
    @MIFWrite
    def VecMagScalarField(self, name, **kwargs):
        """
        Implements: Oxs_VecMagScalarField 
         Mandatory: field <vector field>
          Optional: multiplier <float=1>, offset <float=0>
        """        
        
        field = kwargs["field"]
        if not field in self.vectorFields:
            raise UndefTokenMIFException("field", "vector field", 
                                         list(self.vectorFields))
        self.labelsUsed.add(field)
        multiplier = kwargs.get("multiplier", 1)
        offset = kwargs.get("offset", 0)
        self.scalarFields.add(name)
        
        return ["Specify Oxs_VecMagScalarField:%s {" % name,
                "    field %s" % field,
                "    multiplier %s" % multiplier,
                "    offset %s" % offset,
                "}"]
        
    @MIFWrite
    def ScriptOrientScalarField(self, name, **kwargs):
        """
        Implements: Oxs_ScriptOrientScalarField 
         Mandatory: field <scalar field>, script <script>
                    atlas <atlas> OR xrange <(float, float)>, yrange <(float, float)>, zrange <(float, float)>
          Optional: script_args <["relpt" and/or "rawpt" a/o "minpt" a/o "maxpt" a/o "span"], ["relpt"]>
        """
        
        field = kwargs["field"]
        if not field in self.scalarFields:
            raise UndefTokenMIFException("field", "scalar field", 
                                         list(self.scalarFields))
        self.labelsUsed.add(field)
            
        script = kwargs["script"]
        script_args = kwargs.get("script_args", ["relpt"])
        
        for i,v in enumerate(script_args):
            if not v in set(["relpt", "rawpt", "minpt", "maxpt", "span"]):
                raise MIFException("script_args", 
                                   "%s is not 'relpt', 'rawpt', 'minpt', 'maxpt', or 'span'" % v)
            if v in script_args[:i]:
                raise MIFException("script_args", 
                                   "%s appears more than once in script_args" % v)
            
        retWork = ["Specify Oxs_ScriptOrientScalarField:%s {" % name,
                   "    field %s" % field,
                   "    script %s" % script,
                   "    script_args { %s }" % " ".join(script_args)]
        
        if "atlas" in kwargs:
            atlas = kwargs["atlas"]
            if not atlas in self.atlases:
                raise UndefTokenMIFException("atlas", 
                                             "atlas", list(self.atlases))
            self.labelsUsed.add(atlas)
            retWork.append("    atlas %s" % atlas)
        elif "xrange" in kwargs and "yrange" in kwargs and "zrange" in kwargs:
            xmin, xmax = kwargs["xrange"]
            ymin, ymax = kwargs["yrange"]
            zmin, zmax = kwargs["zrange"]
            quickValidateExtent("ScriptOrientScalarField", name, 
                                xmin, xmax, ymin, ymax, zmin, zmax)
        
            maxSignificance = sigfigs([xmin, xmax, ymin, ymax, zmin, zmax])
            
            retWork.extend(["    xrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(xmin, xmax, sigfigs=maxSignificance),
                            "    yrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(ymin, ymax, sigfigs=maxSignificance),
                            "    zrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(zmin, zmax, sigfigs=maxSignificance)])
        else:
            raise MIFException("atlas or xrange/yrange/zrange", "Didn't specify atlas or all of (xrange, yrange, zrange)")
        
        self.scalarFields.add(name)
        retWork.append("}")
        
    @MIFWrite
    def AffineOrientScalarField(self, name, **kwargs):
        """
        Implements: Oxs_AffineOrientScalarField
         Mandatory: field <scalar field>, M <[] of 1, 3, 6, or 9 floats>
          Optional: offset <(x,y,z)=(0, 0, 0)>, inverse <0 or 1=0>, inverse_slack <int=128>
        """
        field = kwargs["field"]
        if not field in self.scalarFields:
            raise UndefTokenMIFException("field", "scalar field", 
                                         list(self.scalarFields))
        self.labelsUsed.add(field)
        
        M = kwargs["M"]
        
        if not len(M) in [1, 3, 6, 9]:
            raise MIFException("M", "Gave %s elements; must be 1, 3, 6 or 9 elements." % len(M))
        
        offX, offY, offZ = kwargs.get("offset", (0, 0, 0))
        inverse = kwargs.get("inverse", 0)
        inverse_slack = kwargs.get("inverse_slack", 1)
        
        self.scalarFields.add(name)      
        
        return ["Specify Oxs_AffineOrientScalarField:%s {" % name,
                "    field %s" % field,
                "    M { %s }" % " ".join(M),
                "    offset { %s %s %s }" % (offX, offY, offZ),
                "    inverse %s" % inverse,
                "    inverse_slack %s" % inverse_slack,
                "}"]
        
    @MIFWrite
    def AffineTransformScalarField(self, name, **kwargs):
        """
        Implements: Oxs_AffineTransformScalarField 
         Mandatory: field <scalar field>
          Optional: multiplier <float=1>, offset <float=0>, inverse <0 or 1=0>
        """
        field = kwargs["field"]
        if not field in self.scalarFields:
            raise UndefTokenMIFException("field", "scalar field", 
                                         list(self.scalarFields))       
        self.labelsUsed.add(field)

        multiplier = kwargs.get("multiplier", 1)
        offset = kwargs.get("offset", 0)
        inverse = kwargs.get("inverse", 0)
        
        if inverse == 1 and multiplier == 0:
            raise MIFException("inverse/multiplier", "inverting with multiplier = 0 divides by zero")
        
        self.scalarFields.add(name)
        
        return ["Specify Oxs_AffineTransformScalarField:%s {" % name,
                "    field %s" % field,
                "    multiplier %s" % multiplier,
                "    offset %s" % offset,
                "    inverse %s" % inverse,
                "}"]
        
    @MIFWrite
    def ImageScalarField(self, name, **kwargs):
        """
        Implements: Oxs_ImageScalarField 
         Mandatory: image <filename>, viewplane <'xy' or 'zx' or 'yz'>
                    atlas <atlas> OR xrange <(float, float)>, yrange <(float, float)>, zrange <(float, float)>
          Optional: multiplier <float=1>, offset <float=0>, invert <0 or 1=0>
                    exterior <float, 'boundary', or 'error'='error'>
        """
        image = kwargs["image"]
        image = kwargs["image"].replace("\\", "/") # Take care - OOMMF only uses /
        invert = kwargs.get("invert", 0)
        multiplier = kwargs.get("multiplier", 1)
        offset = kwargs.get("offset", 0)
        viewplane = kwargs["viewplane"]
        if not viewplane in ["xy", "zx", "yz"]:
            raise MIFException("viewplane", "viewplane %s is not one of xy, zx, yx" % viewplane)
                
        retWork = ["Specify Oxs_ImageScalarField:%s {" % name,
                   "    image %s" % image,
                   "    invert %s" % invert,
                   "    multipler %s" % multiplier,
                   "    offset %s" % offset,
                   "    viewplane %s" % view]
        
        if "atlas" in kwargs:
            atlas = kwargs["atlas"]
            if not atlas in self.atlases:
                raise UndefTokenMIFException("atlas", "atlas", list(self.atlases))
            self.labelsUsed.add(atlas)
            retWork.append("    atlas %s" % atlas)
        elif "xrange" in kwargs and "yrange" in kwargs and "zrange" in kwargs:
            xmin, xmax = kwargs["xrange"]
            ymin, ymax = kwargs["yrange"]
            zmin, zmax = kwargs["zrange"]
            quickValidateExtent("ScriptOrientScalarField", name, 
                                xmin, xmax, ymin, ymax, zmin, zmax)
        
            maxSignificance =sigfigs([xmin, xmax, ymin, ymax, zmin, zmax])
            
            retWork.extend(["    xrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(xmin, xmax, sigfigs=maxSignificance),
                            "    yrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(ymin, ymax, sigfigs=maxSignificance),
                            "    zrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(zmin, zmax, sigfigs=maxSignificance)])
        else:
            raise MIFException("atlas or xrange/yrange/zrange", "Didn't specify atlas or all of (xrange, yrange, zrange)")
 
        exterior = kwargs.get("exterior", "error")
        if isinstance(exterior, basestring):
            if not exterior in ["boundary", "error"]:
                raise MIFException("exterior", "%s is not a number, 'boundary', or 'error'" % exterior)
        retWork.append("    exterior %s" % exterior)
        
        self.scalarFields.add(name)
        retWork.append("}")
        return retWork
    
    # 
    ###
    ##### VECTOR FIELDS
    ###
    #
    
    @MIFWrite
    def UniformVectorField(self, name, **kwargs):
        """
        Implements: Oxs_UniformVectorField 
         Mandatory: vector <(x, y, z)>
          Optional: norm <float>
        """
        
        x, y, z = kwargs["vector"]
        norm = kwargs.get("norm")
        
        retWork = ["Specify Oxs_UniformVectorField:%s {" % name,
                   "    vector { %s %s %s }" % (x, y, z)]
        if norm:
            retWork.append("    norm %s" % norm)
            
        retWork.append("}")
        self.vectorFields.add(name)
        return retWork
    
    @MIFWrite
    def AtlasVectorField(self, name, **kwargs):
        """
        Implements: Oxs_AtlasVectorField 
         Mandatory: atlas <atlas>, values <[(region, (x, y, z)), (region (x, y, z)), ... ]>
          Optional: norm <float>, multiplier <float=1>, default_value <(x, y, z)>
        """
        
        atlas = kwargs["atlas"]
        if not atlas in self.atlases:
            raise UndefTokenMIFException("atlas", "atlas", list(self.atlases))
        self.labelsUsed.add(atlas)
        
        norm = kwargs.get("norm")
        multiplier = kwargs.get("multiplier", 1)
        default_value = kwargs.get("default_value")
        if default_value and isinstance(default_value, basestring):
            defaultIsField = True
            if not default_value in self.vectorFiends:
                raise UndefTokenMIFException("default_value", "vector field", 
                                             list(self.vectorFields))
            self.labelsUsed.add(default_value)
        elif default_value:
            defaultIsField = False
        
        values = kwargs["values"]
        for region in map(lambda x: x[0], values):
            if not region in self.regions[atlas]:
                raise UndefTokenMIFException("region", "region", 
                                             list(self.regions[atlas]))
            
        retWork = ["Specify Oxs_AtlasVectorField:%s {" % name,
                   "    atlas %s" % atlas,
                   "    multiplier %s" % multiplier]
        
        if norm:
            retWork.append("    norm %s" % norm)
        if default_value: 
            if defaultIsField:
                retWork.append("    default_value %s" % default_value)
            else:
                retWork.append("    default_value {{ {0[0]} {0[1]} {0[2]} }}".format(default_value))
                
        retWork.append("    values {")
        
        maxSignificance = max(map(lambda x: max(map(lambda y: len(str(y)), 
                                                    x[1])), values))
        
        for region, (x, y, z) in values:
            retWork.append("    {0} {{ {1:< .{sigfigs}e} {2:< .{sigfigs}e} {3:< .{sigfigs}e} }}".format(region, x, y, z, sigfigs=maxSignificance))
            
        retWork.extend(["    }", "}"])
        self.vectorFields.add(name)
        return retWork
    
    @MIFWrite
    def Proc(self, name, **kwargs):
        """
        Implements: Tcl process
         Mandatory: args <[string, string, ...]>, lines <[string, string, ...]>
          Optional: None
        """
        args = kwargs["args"]
        argUsage = defaultdict(lambda: False)
        lines = kwargs["lines"]
        
        for line in lines:
            for arg in args:
                if arg in line:
                    argUsage[arg] = True
        for arg in args:
            if not argUsage[arg]:
                print("(%s) Warning: %s taken as script argument but never referenced." % (self.filename, arg))
        
        hasReturned = False
        for line in lines:
            if "return" in line:
                hasReturned = True
                break
        if not hasReturned:
            raise MIFException("lines", "The specified script does not return anything.")
        
        self.scripts.add(name)
        retWork = ["proc %s { %s } {" % (name, " ".join(args))]
        retWork.extend(["    " + line for line in lines])
        retWork.append("}")
        return retWork
    
    @MIFWrite
    def ScriptVectorField(self, name, **kwargs):
        """
        Implements: Oxs_ScriptVectorField
         Mandatory: script <script>
          Optional: script_args <["relpt" and/or "rawpt" a/o "minpt" a/o "maxpt" a/o "span"
                                  a/o "scalar_fields" a/o "vector_fields"], ["relpt"]>
                    atlas <atlas> OR xrange <(float, float)>, yrange <(float, float)>, zrange <(float, float)>
                    scalar_fields <[field, field, ... ]>, vector_fields <[field, field, ... ]>
                    norm <float>, multiplier <float=1>
        """
        script = kwargs["script"]
        if not script in self.scripts:
            raise UndefTokenMIFException("script", "script", list(self.scripts))
        script_args = kwargs.get("script_args", ["relpt"])
        
        norm = kwargs.get("norm")
        multiplier = kwargs.get("multiplier")
        
        for i, v in enumerate(script_args):
            if not v in set(["relpt", "rawpt", "minpt", "maxpt", "span", "scalar_fields", "vector_fields"]):
                raise MIFException("script_args", "%s is not 'relpt', 'rawpt', 'minpt', 'maxpt', or 'span'" % v)
            if v in script_args[:i]:
                raise MIFException("script_args", 
                                   "%s appears more than once in script_args" % v)
            
        retWork = ["Specify Oxs_ScriptVectorField:%s {" % name,
                   "    script %s" % script]
        
        if norm:
            retWork.append("    norm %s" % norm)
        
        if not "scalar_fields" in script_args and not "vector_fields" in script_args:
            retWork.append("    script_args { %s }" % " ".join(script_args))
        else:
            retWork.append("    script_args {")
            # Can't reorder, so can't do a list comp
            for label in script_args:
                if label not in ["scalar_fields", "vector_fields"]:
                    retWork.append("        %s" % label)
                elif label == "scalar_fields":
                    if not "scalar_fields" in kwargs:
                        raise MIFException("scalar_fields", "Used scalar_fields in script_args, but didn't supply any")
                    scalar_fields = kwargs["scalar_fields"]
                    retWork.append("        scalar_fields {")
                    for scalar_field in scalar_fields:
                        if scalar_field not in self.scalarFields:
                            raise UndefTokenMIFException("scalar_field", 
                                                         "scalar field", 
                                                         list(self.scalarFields))
                        retWork.append("            %s" % scalar_field)
                        self.labelsUsed.add(scalar_field)
                    retWork.append("        }")
                elif label == "vector_fields":
                    if not "vector_fields" in kwargs:
                        raise MIFException("vector_fields", "Used vector_fields in script_args, but didn't supply any")
                    vector_fields = kwargs["vector_fields"]
                    retWork.append("        vector_fields {")
                    for vector_field in vector_fields:
                        if vector_field not in self.scalarFields:
                            raise UndefTokenMIFException("vector_field", 
                                                         "vector field", 
                                                         list(self.scalarFields))
                        retWork.append("            %s" % vector_field)
                        self.labelsUsed.add(vector_field)
                    retWork.append("        }")                    
            retWork.append("    }")
        
        if "atlas" in kwargs:
            atlas = kwargs["atlas"]
            if not atlas in self.atlases:
                raise UndefTokenMIFException("atlas", "atlas", 
                                             list(self.atlases))
            self.labelsUsed.add(atlas)
            retWork.append("    atlas %s" % atlas)
        elif "xrange" in kwargs and "yrange" in kwargs and "zrange" in kwargs:
            xmin, xmax = kwargs["xrange"]
            ymin, ymax = kwargs["yrange"]
            zmin, zmax = kwargs["zrange"]
            quickValidateExtent("ScriptVectorField", name, 
                                xmin, xmax, ymin, ymax, zmin, zmax)
        
            maxSignificance = sigfigs([xmin, xmax, ymin, ymax, zmin, zmax])
            
            retWork.extend(["    xrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(xmin, xmax, sigfigs=maxSignificance),
                            "    yrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(ymin, ymax, sigfigs=maxSignificance),
                            "    zrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(zmin, zmax, sigfigs=maxSignificance)])
        else:
            raise MIFException("atlas or xrange/yrange/zrange", "Didn't specify atlas or all of (xrange, yrange, zrange)")
        
        self.vectorFields.add(name)
        retWork.append("}")
        return retWork
    
    @MIFWrite
    def FileVectorField(self, name, **kwargs):
        """
        Implements: Oxs_FileVectorField 
         Mandatory: file <filename> 
          Optional: { atlas <atlas> OR xrange <(float, float)>, yrange <(float, float)>, zrange <(float, float)> }
                    OR
                    { spatial_scaling <(x, y, z)>, spatial_offset <(x, y, z)> }
                    norm <float>, multiplier <float=1>, exterior <float, 'boundary', or 'error'='error>
        """
        
        file = kwargs["file"]
        self.referencedFilenames.add(file)
        
        norm = kwargs.get("norm")
        multiplier = kwargs.get("multiplier", 1)
        
        retWork = ["Specify Oxs_FileVectorField:%s {" % name,
                   "    file %s" % file,
                   "    multiplier %s" % multiplier]
        
        if norm:
            retWork.append("    norm %s" % norm)
        
        # Implicit and explicit line continuations at the SAME TIME!
        if ("atlas" in kwargs or "xrange" in kwargs or 
            "yrange" in kwargs or "zrange" in kwargs) and \
           ("spatial_scaling" in kwargs or "spatial_offset" in kwargs):
            raise MIFException("everything", "Specified atlas/?range bounding AND spatial_scaling/spatial_offset bounding")
        
        if "spatial_scaling" in kwargs and "spatial_offset" in kwargs:
            spatial_scaling = kwargs["spatial_scaling"]
            spatial_offset = kwargs["spatial_offset"]
            retWork.extend(["    spatial_offset {{ {0[0]} {0[1]} {0[2]} }}".format(spatial_offset),
                            "    spatial_scaling {{ {0[0]} {0[1]} {0[2]} }}".format(spatial_scaling)])
        
        elif "atlas" in kwargs:
            atlas = kwargs["atlas"]
            if not atlas in self.atlases:
                raise UndefTokenMIFException("atlas", "atlas", list(self.atlases))
            self.labelsUsed.add(atlas)
            retWork.append("    atlas %s" % atlas)
        elif "xrange" in kwargs and "yrange" in kwargs and "zrange" in kwargs:
            xmin, xmax = kwargs["xrange"]
            ymin, ymax = kwargs["yrange"]
            zmin, zmax = kwargs["zrange"]
            quickValidateExtent("FileVectorField", name, xmin, xmax, ymin, ymax, zmin, zmax)
        
            maxSignificance = sigfigs([xmin, xmax, ymin, ymax, zmin, zmax])
            
            retWork.extend(["    xrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(xmin, xmax, sigfigs=maxSignificance),
                            "    yrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(ymin, ymax, sigfigs=maxSignificance),
                            "    zrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(zmin, zmax, sigfigs=maxSignificance)])
        else:
            raise MIFException("everything", "Didn't specify atlas or all of (xrange, yrange, zrange) or all of (spatial_scaling, spatial_offset")
 
 
        exterior = kwargs.get("exterior", "error")
        if isinstance(exterior, basestring):
            if not exterior in ["boundary", "error"]:
                raise MIFException("exterior", "%s is not a number, 'boundary', or 'error'" % exterior)
        retWork.append("    exterior %s" % exterior)
        retWork.append("}")
        self.vectorFields.add(name)
        return retWork
    
    @MIFWrite
    def RandomVectorField(self, name, **kwargs):
        """
        Implements: Oxs_RandomVectorField
         Mandatory: min_norm <float>, max_norm <float>
          Optional: cache_grid <mesh>
        """
        
        min_norm = kwargs["min_norm"]
        max_norm = kwargs["max_norm"]
        
        if max_norm < min_norm:
            raise MIFException("min_norm, max_norm", "minimum %s greater than maximum %s" % (min_norm, max_norm))
        
        retWork = ["Specify Oxs_RandomVectorField:%s {" % name,
                   "    min_norm %s" % min_norm,
                   "    max_norm %s" % max_norm]
        
        if "cache_grid" in kwargs:
            cache_grid = kwargs["cache_grid"]
            if not cache_grid in self.meshes:
                raise UndefTokenMIFException("cache_grid", "mesh", 
                                             list(self.meshes))
            retWork.append("    cache_grid %s" % cache_grid)
            
        retWork.append("}")
        if name:
            self.vectorFields.add(name)
        return retWork
    
    @MIFWrite
    def PlaneRandomVectorField(self, name, **kwargs):
        """
        Implements: Oxs_PlaneRandomVectorField
         Mandatory: min_norm <float>, max_norm <float>, plane_normal <(x, y, z)>
          Optional: cache_grid <mesh>
        """
        
        retWork = ["Specify Oxs_PlaneRandomVectorField:%s {" % name]
        
        plane_normal = kwargs["plane_normal"]
        retWork.append("    plane_normal {{ {0[0]} {0[1]} {0[2]} }}".format(plane_normal))
        
        retWork.extend( self.RandomVectorField(_noWrite = True, _noName = True, 
                                               **kwargs)[1:-1] )
            
        retWork.append("}")
        self.vectorFields.add(name)
        return retWork    
    
    @MIFWrite
    def ScriptOrientVectorField(self, name, **kwargs):
        """
        Implements: Oxs_ScriptOrientVectorField 
         Mandatory: field <vector field>, script <script>
                    atlas <atlas> OR xrange <(float, float)>, yrange <(float, float)>, zrange <(float, float)>
          Optional: script_args <["relpt" and/or "rawpt" a/o "minpt" a/o "maxpt" a/o "span"], ["relpt"]>
        """
        
        field = kwargs["field"]
        if not field in self.vectorFields:
            raise UndefTokenMIFException("field", "vector field", 
                                         list(self.vectorFields))
        self.labelsUsed.add(field)
            
        script = kwargs["script"]
        if not script in self.scripts:
            raise UndefTokenMIFException("script", "script", list(self.scripts))        
        script_args = kwargs.get("script_args", ["relpt"])
        
        for i,v in enumerate(script_args):
            if not v in set(["relpt", "rawpt", "minpt", "maxpt", "span"]):
                raise MIFException("script_args", "%s is not 'relpt', 'rawpt', 'minpt', 'maxpt', or 'span'" % v)
            if v in script_args[:i]:
                raise MIFException("script_args", "%s appears more than once in script_args" % v)
            
        retWork = ["Specify Oxs_ScriptOrientVectorField:%s {" % name,
                   "    field %s" % field,
                   "    script %s" % script,
                   "    script_args { %s }" % " ".join(script_args)]
        
        if "atlas" in kwargs:
            atlas = kwargs["atlas"]
            if not atlas in self.atlases:
                raise UndefTokenMIFException("atlas", "atlas", 
                                             list(self.atlases))
            self.labelsUsed.add(atlas)
            retWork.append("    atlas %s" % atlas)
        elif "xrange" in kwargs and "yrange" in kwargs and "zrange" in kwargs:
            xmin, xmax = kwargs["xrange"]
            ymin, ymax = kwargs["yrange"]
            zmin, zmax = kwargs["zrange"]
            quickValidateExtent("ScriptOrientVectorField", name, 
                                xmin, xmax, ymin, ymax, zmin, zmax)
        
            maxSignificance = sigfigs([xmin, xmax, ymin, ymax, zmin, zmax])
            
            retWork.extend(["    xrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(xmin, xmax, sigfigs=maxSignificance),
                            "    yrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(ymin, ymax, sigfigs=maxSignificance),
                            "    zrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(zmin, zmax, sigfigs=maxSignificance)])
        else:
            raise MIFException("atlas or xrange/yrange/zrange", "Didn't specify atlas or all of (xrange, yrange, zrange)")
        
        self.vectorFields.add(name)
        retWork.append("}")
        
    @MIFWrite
    def AffineOrientVectorField(self, name, **kwargs):
        """
        Implements: Oxs_AffineOrientVectorField
         Mandatory: field <vector field>, M <[] of 1, 3, 6, or 9 floats>
          Optional: offset <(x,y,z)=(0, 0, 0)>, inverse <0 or 1=0>, inverse_slack <int=128>
        """
        field = kwargs["field"]
        if not field in self.vectorFields:
            raise UndefTokenMIFException("field", "vector field", 
                                         list(self.vectorFields))
        self.labelsUsed.add(field)
        
        M = kwargs["M"]
        
        if not len(M) in [1, 3, 6, 9]:
            raise MIFException("M", "Gave %s elements; must be 1, 3, 6 or 9 elements." % len(M))
        
        offX, offY, offZ = kwargs.get("offset", (0, 0, 0))
        inverse = kwargs.get("inverse", 0)
        inverse_slack = kwargs.get("inverse_slack", 1)
        
        self.vectorFields.add(name)      # self.scalarFields.add(name)     # changed by Sumit Dutta on 10/21/2013 due to bug
        
        return ["Specify Oxs_AffineOrientVectorField:%s {" % name,
                "    field %s" % field,
                "    M { %s }" % " ".join(M),
                "    offset { %s %s %s }" % (offX, offY, offZ),
                "    inverse %s" % inverse,
                "    inverse_slack %s" % inverse_slack,
                "}"]
        
    @MIFWrite
    def AffineTransformScalarField(self, name, **kwargs):
        """
        Implements: Oxs_AffineTransformScalarField 
         Mandatory: field <scalar field>
          Optional: multiplier <float=1>, offset <float=0>, inverse <0 or 1=0>
        """
        field = kwargs["field"]
        if not field in self.scalarFields:
            raise UndefTokenMIFException("field", "scalar field", 
                                         list(self.scalarFields))       
        self.labelsUsed.add(field)

        multiplier = kwargs.get("multiplier", 1)
        offset = kwargs.get("offset", 0)
        inverse = kwargs.get("inverse", 0)
        
        if inverse == 1 and multiplier == 0:
            raise MIFException("inverse/multiplier", "inverting with multiplier = 0 divides by zero")
        
        self.scalarFields.add(name)
        
        return ["Specify Oxs_AffineTransformScalarField:%s {" % name,
                "    field %s" % field,
                "    multiplier %s" % multiplier,
                "    offset %s" % offset,
                "    inverse %s" % inverse,
                "}"]
        
    @MIFWrite
    def MaskVectorField(self, name, **kwargs):
        """
        Implements: Oxs_MaskVectorField 
         Mandatory: mask <scalar field>, field <vector field>
          Optional: None
        """
        
        mask = kwargs["mask"]
        if not mask in self.scalarFields:
            raise UndefTokenMIFException("mask", "scalar field", 
                                         list(self.scalarFields))
        
        field = kwargs["field"]
        if not field in self.vectorFields:
            raise UndefTokenMIFException("field", "vector field", 
                                         list(self.vectorFields))
        
        self.vectorFields.add(name)
        return ["Specify Oxs_MaskVectorField:%s {" % name,
                "    mask %s" % mask,
                "    field %s" % field,
                "}"]
        
    @MIFWrite
    def ImageVectorField(self, name, **kwargs):
        """
        Implements: Oxs_ImageVectorField
         Mandatory: image <filename>, viewplane <'xy' or 'zx' or 'yz'>
                    atlas <atlas> OR xrange <(float, float)>, yrange <(float, float)>, zrange <(float, float)>
          Optional: multiplier <float=1>, offset <float=0>, invert <0 or 1=0>
                    exterior <float, 'boundary', or 'error'='error'>
        """
        image = kwargs["image"]
        image = kwargs["image"].replace("\\", "/") # Take care - OOMMF only uses /
        invert = kwargs.get("invert", 0)
        multiplier = kwargs.get("multiplier", 1)
        offset = kwargs.get("offset", 0)
        viewplane = kwargs["viewplane"]
        if not viewplane in ["xy", "zx", "yz"]:
            raise MIFException("viewplane", 
                               "viewplane %s is not one of xy, zx, yx" % viewplane)
                
        retWork = ["Specify Oxs_ImageVectorField:%s {" % name,
                   "    image %s" % image,
                   "    invert %s" % invert,
                   "    multipler %s" % multiplier,
                   "    offset %s" % offset,
                   "    viewplane %s" % view]
        
        if "atlas" in kwargs:
            atlas = kwargs["atlas"]
            if not atlas in self.atlases:
                raise UndefTokenMIFException("atlas", "atlas", list(self.atlases))
            self.labelsUsed.add(atlas)
            retWork.append("    atlas %s" % atlas)
        elif "xrange" in kwargs and "yrange" in kwargs and "zrange" in kwargs:
            xmin, xmax = kwargs["xrange"]
            ymin, ymax = kwargs["yrange"]
            zmin, zmax = kwargs["zrange"]
            quickValidateExtent("ScriptOrientScalarField", name, 
                                xmin, xmax, ymin, ymax, zmin, zmax)
        
            maxSignificance = sigfigs([xmin, xmax, ymin, ymax, zmin, zmax])
            
            retWork.extend(["    xrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(xmin, xmax, sigfigs=maxSignificance),
                            "    yrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(ymin, ymax, sigfigs=maxSignificance),
                            "    zrange {{ {0:< .{sigfigs}e} {1:< .{sigfigs}e} }}".format(zmin, zmax, sigfigs=maxSignificance)])
        else:
            raise MIFException("atlas or xrange/yrange/zrange", "Didn't specify atlas or all of (xrange, yrange, zrange)")
 
        exterior = kwargs.get("exterior", "error")
        if isinstance(exterior, basestring):
            if not exterior in ["boundary", "error"]:
                raise MIFException("exterior", 
                                   "%s is not a number, 'boundary', or 'error'" % exterior)
        retWork.append("    exterior %s" % exterior)
        
        self.vectorFields.add(name)
        retWork.append("}")
        return retWork
    
    # 
    ###
    ##### SCHEDULES AND DESTINATIONS
    ###
    #    
    
    @NonBlock
    @NoPad
    @MIFWrite
    def Destination(self, name, **kwargs):
        """
        Implements: Destination 
         Mandatory: label <string>, type <'mmDisp' or 'mmGraph' or 'mmArchive' or 'mmDataTable'
          Optional: new <True or False=False>
           WARNING: It is not currently possible to validate the data types being linked in this way.
        """
        
        label = kwargs["label"]
        ptype = kwargs["type"]
        new = kwargs.get("new")
        
        if not ptype in ["mmDisp", "mmGraph", "mmArchive", "mmDataTable"]:
            raise MIFException("type", 
                               "%s is not one of %s" % (ptype, 
                                                        ["mmDisp", "mmGraph", 
                                                         "mmArchive", 
                                                         "mmDataTable"]),
                               "Destination %s" % ptype, label)
        
        self.destinations.add(label)
        retWork = "Destination %s %s" % (label, ptype)
        if new:
            retWork += " new"
        
        return [retWork]
    
    @NonBlock
    @NoPad
    @MIFWrite
    def Schedule(self, name, **kwargs):
        """
        Implements: Schedule 
         Mandatory: output <an OOMMF output>, label <destination>
                    stage <int> AND/OR step <int>
          Optional: None
        """        
        
        # TODO: Track outputs! They are known by class...
        # TODO: Accept unnamed outputs and collect the names
        
        output = kwargs["output"]
        label = kwargs["label"]
        
        if not label in self.destinations:
            raise UndefTokenMIFException("label", "destination", 
                                         list(self.destinations))
        
        stage = kwargs.get("stage")
        step = kwargs.get("step")
        
        retWork = []
        
        if stage:
            retWork.append("Schedule %s %s Stage %s" % (output, label, stage))
        if step:
            retWork.append("Schedule %s %s Step %s" % (output, label, step))
            
        self.scheduled = True
        return retWork