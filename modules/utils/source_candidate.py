############################################################
# Helper class to store source candidate information
############################################################

class SourceCandidate:
    def __init__(self, i, j, k = None, kslice = None, sig = None):
        self.i = i
        self.j = j
        self.k = k
        self.sig = sig
        self.kslice = kslice

        # default setting for central frequency index and source significance
        if kslice != None:
            if k == None: self.k = int(kslice.start *0.5 + kslice.stop*0.5)
        #if sig == None: sig = kslice.stop - kslice.start        

    def matching(self, truth_source, margin = 10):
        # if truth_source.z() >= self.kslice.start and truth_source.z() <= self.kslice.stop:
            # print(T.tcol("Is matching: {} <= {} <= {}".format(self.kslice.start, truth_source.z(), self.kslice.stop), "yellow"))
        # else:
            # print(T.tcol("Not matching: {} <= {} <= {}".format(self.kslice.start, truth_source.z(), self.kslice.stop), "blue"))
        # print(T.tcol("Are both matching ? {2} >= {0} is {3}, {2} <= {1} is {4}".format(self.kslice.start, self.kslice.stop, truth_source.z(), truth_source.z() >= self.kslice.start, truth_source.z() <= self.kslice.stop), "yellow"))
        return truth_source.z() >= self.kslice.start-margin and truth_source.z() <= self.kslice.stop + margin
    def __str__(self):
        return "Source candidate at i = {0}, j = {1}, k = {2}, significance = {3}".format(self.i,self.j,self.k, self.sig)
    def to_line(self):
        return "{0} {1} {2} {3}\n".format(self.i,self.j,self.k, self.sig)

############################################################
# Source grouping helper class and methods
############################################################

class SourceGroup:
    def __init__(self, source):
        self._source_list = [source]
        self._max = source
        self._range_min = source[:-1]
        self._range_max = source[:-1]
    def __iadd__(self,other):
        self._source_list.append(other)
        for i in range(3):
            if other[i] < self._range_min[i]:
                self._range_min[i] = other[i]
            if other[i] > self._range_max[i]:
                self._range_max[i] = other[i]

        if other[-1] > self._max[-1]:
            self._max = other
    def merge(self, other):
        for s in other._source_list:
            if s not in self._source_list:
                self.__iadd__(s)
    def __str__(self):
        return "{0}-{1}-{2}; {3}-{4}-{5}; {6}-{7}-{8};".format(self._range_min[0], self._max[0], self._range_max[0],
                                      self._range_min[1], self._max[1], self._range_max[1],
                                      self._range_min[2], self._max[2], self._range_max[2])

    # check if a source is adjancent to any other source in the group
    def source_is_adjacent(self, other, dvox = 10):
        x0, y0, z0, sig0 = other
        for x, y, z, sig in self._source_list:
            if abs(x-x0) <= dvox and abs(y-y0) <= dvox and abs(z-z0) <= dvox*2:
                return True
        return False

    def overlaps(self, other):
        groups_overlap = True
        for i in range(3):
            overlapping_range = range(max(self._range_min[i], other._range_min[i]), min(self._range_max[i], other._range_max[i])+1)
            groups_overlap &= len(overlapping_range) > 0
        return groups_overlap

    @property
    def center(self):
        return self._max

    @property
    def symmetryx(self):
        side1 = self.center[0] - self._range_min[0]
        side2 = self._range_max[0] - self.center[0]
        return side1 - side2

    @property
    def symmetryy(self):
        side1 = self.center[1] - self._range_min[1]
        side2 = self._range_max[1] - self.center[1]
        return side1 - side2

    @property
    def symmetryz(self):
        side1 = self.center[2] - self._range_min[2]
        side2 = self._range_max[2] - self.center[2]
        return side1 - side2


    @property
    def size(self):
        return len(self._source_list)

    @property
    def lenx(self):
        return self._range_max[0] - self._range_min[0]

    @property
    def leny(self):
        return self._range_max[1] - self._range_min[1]

    @property
    def lenz(self):
        return self._range_max[2] - self._range_min[2]

def print_overlaps(groups):
    for g in groups:
        print(g)
        for g2 in groups:
            if g != g2 and g.overlaps(g2):
                print("  overlapping:", g2)

def prune_candidates(infile, outfile, k):
    sources = []
    with open(infile, 'r') as f:
        for line in f:
            x, y, z, sig = line.split()
            if int(z) == k: continue
            sources.append([ int(x), int(y), int(z), float(sig)])

    with open(outfile, 'w') as f:
        for s in sources:
            f.write( "{0} {1} {2} {3}\n".format(*s))

def count_neighboring_groups(x1, y1, others, d = 30):
    neighbors = 0
    for x2, y2 in others:
        if x1 == x2 and y1 == y2: continue
        dcel = math.sqrt( (x1 - x2)**2 + (y1 - y2)**2)
        if dcel < d: neighbors += 1
    return neighbors



def remove_suspicious_candidates(infile, outfile, dvox = 30, max_neighbors = 3):
    sources = []
    with open(infile, 'r') as f:
        for line in f:
            x, y, z, sig = line.split()
            sources.append([ int(x), int(y), int(z), float(sig)])
    keep_sources = []
    for s1 in sources:
        n_line_neighbors = count_neighboring_groups(s1[0], s1[1], [(s2[0], s2[1]) for s2 in sources], d = dvox)
        '''for s2 in sources:
            if s1 == s2: continue
            dcel = math.sqrt( (s1[0] - s2[0])**2 + (s1[1] - s2[1])**2)
            if dcel < dvox:
                n_line_neighbors += 1

        '''
        if n_line_neighbors <= max_neighbors: keep_sources.append(s1)

    with open(outfile, 'w') as f:
        for s in keep_sources:
            f.write( "{0} {1} {2} {3}\n".format(*s))

def make_source_groups(infile, dvox,  min_sig_threshold = -999):
    groups = []
    # assign each source to groups
    nsources = 0
    with open(infile, 'r') as f:
        for line in f:
            
            x, y, z, sig = line.split()
            source = [ int(x), int(y), int(z), float(sig)]
            if source[-1] < min_sig_threshold: continue

            nsources += 1

            found_group = False
            for g in groups:
                if g.source_is_adjacent(source, dvox):
                    g += source
                    found_group = True

            if not found_group:
                groups.append( SourceGroup(source))

    # merge overlapping groups
    for i in range(len(groups)):
        for j in range(len(groups)):
            if groups[i] == None or groups[j] == None or groups[i] == groups[j]: continue
            if groups[i].overlaps(groups[j]):
                groups[i].merge(groups[j])
                groups[j] = None
    groups = [g for g in groups if g != None]

    # make sure that we kept all sources
    assert nsources == sum([g.size for g in groups])
    return groups
# only keep sources in groups with group size >= min_group_size OR central significance > pasing_sig_threshold
def merge_source_candidates(infile, outfile, dvox = 10, min_group_size = 1, min_sig_threshold = -999, passing_sig_threshold = -999, verbose = False):

    groups = make_source_groups(infile, dvox,  min_sig_threshold)
    group_filter = lambda g:  g.size >= min_group_size and (True if passing_sig_threshold < 0 else g.center[-1] > passing_sig_threshold)
    groups = [g for g in groups if g != None and group_filter(g)]

    with open(outfile, 'w') as f:
        for g in groups:
            if verbose: print("Group with {0} sources centered at ({1}, {2}, {3}, sig = {4})".format(g.size, *g.center))
            f.write( "{0} {1} {2} {3}\n".format(*g.center))
    #print("#############")
    #print_overlaps(groups)

