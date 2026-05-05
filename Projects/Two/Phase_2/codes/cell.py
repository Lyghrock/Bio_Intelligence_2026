"""Small teaching wrapper around the Hay et al. layer 5 pyramidal cell model.

This file intentionally keeps only the utilities used by the dendritic
computation assignment:

* load the L5 pyramidal-cell morphology and biophysics;
* inspect soma, basal dendrite, apical dendrite, and axon domains;
* select segments by path distance from the soma;
* optionally add passive spines and AMPA/NMDA synapses.

NEURON terms used below:
* Section: one cable branch, e.g. soma[0], dend[12], apic[50].
* Segment: a computational point on a section, e.g. apic[50](0.5).
* x: relative position on a section, from 0 to 1.
"""

from pathlib import Path

import numpy as np
from neuron import h, gui  # gui is imported so NEURON's standard GUI tools are available.


DOMAIN_ALIASES = {
    "soma": "somatic",
    "somatic": "somatic",
    "basal": "basal",
    "dend": "basal",
    "apical": "apical",
    "apic": "apical",
    "axon": "axonal",
    "axonal": "axonal",
    "all": "all",
}


def hoc_path(path):
    """Return a Windows-safe path string for HOC load_file/xopen calls."""

    return Path(path).as_posix()


def configure_synaptic_delayes(jitter_time, con_list, rd, config, cluster_type=None):
    """Set delays for the AMPA/NMDA NetCons created by add_synapses_on_list_of_segments.

    The original assignment code uses this misspelled function name, so we keep
    it for backward compatibility. Each synaptic site has two NetCons: one AMPA
    and one NMDA. They should receive the same presynaptic event time.

    Parameters
    ----------
    jitter_time : float
        Maximum delay in ms. If 0, all synapses are synchronous.
    con_list : list[h.NetCon]
        First half contains AMPA NetCons; second half contains NMDA NetCons.
    rd : h.Random
        NEURON random generator used for optional jitter.
    config : object
        Uses config.CLUSTER_SIZE when cluster_type == "sync_clusters".
    cluster_type : None | "async_clusters" | "sync_clusters"
        None and "async_clusters" jitter each synapse independently. The
        "sync_clusters" option gives each cluster one shared delay.
    """

    n_synapses = len(con_list) // 2
    if n_synapses == 0:
        return

    if jitter_time == 0:
        for con in con_list:
            con.delay = 0
        return

    if cluster_type == "sync_clusters":
        cluster_size = int(config.CLUSTER_SIZE)
        for cluster_start in range(0, n_synapses, cluster_size):
            delay = rd.uniform(0, jitter_time)
            for j in range(cluster_start, min(cluster_start + cluster_size, n_synapses)):
                con_list[j].delay = delay
                con_list[n_synapses + j].delay = delay
    else:
        for j in range(n_synapses):
            delay = rd.uniform(0, jitter_time)
            con_list[j].delay = delay
            con_list[n_synapses + j].delay = delay


class HPC:
    """Convenience class for the human L5 pyramidal-cell assignment model."""

    def __init__(
        self,
        filepath="./",
        spine_density=0.0,
        morphology="cell1.asc",
        biophys="L5PCbiophys3.hoc",
    ):
        """Load morphology, active conductances, and optional spine bookkeeping.

        Parameters
        ----------
        filepath : str | Path
            Project root containing ``morphologies/`` and ``models/``.
        spine_density : float
            Approximate number of spines per um for distal dendritic segments.
            The assignment uses 0.05. Set 0 to keep the model spine-free.
        morphology : str
            Morphology file inside ``morphologies/``.
        biophys : str
            Biophysics file inside ``models/``. The default is biophys3, which
            is the BAC-firing/current-step model requested for this assignment.
        """

        self.root = Path(filepath).resolve()
        self.spine_density = spine_density
        self.HCell = self._load_l5pc_template(morphology, biophys)

        # Passive properties used for the simple spine neck/head compartments.
        self.E_PAS = -86
        self.CM = 0.44
        self.RM = 48300
        self.RA = 261.97

        self.rng = np.random.default_rng(100)
        self.synlist = []
        self.conlist = []
        self.spine_num = 0
        self.spines = []
        self.spine_segments = []
        self.segment_to_spines = {}
        self.segment_to_next_spine = {}

        if spine_density > 0:
            self._prepare_spine_targets(min_distance_um=60)

    def _load_l5pc_template(self, morphology, biophys):
        """Load HOC templates and instantiate the L5 pyramidal cell."""

        morphology_file = self.root / "morphologies" / morphology
        biophys_file = self.root / "models" / biophys
        template_file = self.root / "models" / "L5PCtemplate.hoc"

        if not morphology_file.exists():
            raise FileNotFoundError(morphology_file)
        if not biophys_file.exists():
            raise FileNotFoundError(biophys_file)
        if not template_file.exists():
            raise FileNotFoundError(template_file)

        h.load_file("import3d.hoc")
        h.load_file(hoc_path(biophys_file))
        h.load_file(hoc_path(template_file))
        return h.L5PCtemplate(hoc_path(morphology_file))

    # ------------------------------------------------------------------
    # Domain and distance helpers for teaching morphology access.
    # ------------------------------------------------------------------

    def sections(self, domain="all"):
        """Return NEURON sections from one anatomical domain.

        Examples
        --------
        ``cell.sections("basal")`` returns basal dendritic sections.
        ``cell.sections("apical")`` returns apical dendritic sections.
        """

        canonical = DOMAIN_ALIASES.get(domain)
        if canonical is None:
            valid = ", ".join(sorted(DOMAIN_ALIASES))
            raise ValueError(f"Unknown domain {domain!r}. Use one of: {valid}")
        return list(getattr(self.HCell, canonical))

    def segments(self, domain="all"):
        """Return all computational segments in a domain."""

        return [seg for sec in self.sections(domain) for seg in sec]

    def distance_from_soma(self, seg):
        """Path distance in um from soma[0](0.5) to ``seg``.

        NEURON distance is path distance along the reconstructed morphology, not
        straight-line Euclidean distance.
        """

        h.distance(0, 0.5, sec=self.HCell.soma[0])
        return h.distance(seg.x, sec=seg.sec)

    def segments_in_distance_range(self, domain="all", min_um=0, max_um=None):
        """Select segments by anatomical domain and soma path distance.

        This is the main helper students can use for questions like:
        "find apical dendrite segments 600-800 um away from the soma".
        """

        selected = []
        for seg in self.segments(domain):
            dist = self.distance_from_soma(seg)
            if dist < min_um:
                continue
            if max_um is not None and dist > max_um:
                continue
            selected.append(seg)
        return selected

    def nearest_segment(self, domain="all", target_distance_um=0):
        """Return the segment in a domain closest to a requested soma distance."""

        candidates = self.segments(domain)
        if not candidates:
            raise ValueError(f"No segments found in domain {domain!r}")
        return min(candidates, key=lambda seg: abs(self.distance_from_soma(seg) - target_distance_um))

    def set_channel_density(self, domain, variable_name, value, min_um=0, max_um=None):
        """Set a mechanism variable on segments selected by domain and distance.

        Example
        -------
        Remove fast sodium channels from proximal apical dendrites:

        ``cell.set_channel_density("apical", "gNaTa_tbar_NaTa_t", 0, max_um=150)``

        Returns the number of segments changed. This makes it easy to print a
        sanity check before running a simulation.
        """

        n_changed = 0
        for seg in self.segments_in_distance_range(domain, min_um, max_um):
            if hasattr(seg, variable_name):
                setattr(seg, variable_name, value)
                n_changed += 1
        return n_changed

    def record_segment_variable(self, seg, variable_name):
        """Record a segment variable into an h.Vector.

        ``variable_name`` should omit ``_ref_``. For example:
        ``cell.record_segment_variable(seg, "v")`` or
        ``cell.record_segment_variable(seg, "h_NaTa_t")``.
        """

        ref_name = f"_ref_{variable_name}"
        if not hasattr(seg, ref_name):
            raise AttributeError(f"{seg} has no recordable variable {variable_name!r}")
        return h.Vector().record(getattr(seg, ref_name))

    def summarize_domains(self):
        """Print section/segment counts and distance ranges for each domain."""

        print("Domain summary")
        print("--------------")
        for domain in ["somatic", "basal", "apical", "axonal"]:
            sections = self.sections(domain)
            segments = self.segments(domain)
            distances = [self.distance_from_soma(seg) for seg in segments]
            if distances:
                dist_text = f"{min(distances):.1f}-{max(distances):.1f} um"
            else:
                dist_text = "n/a"
            print(
                f"{domain:8s}: {len(sections):3d} sections, "
                f"{len(segments):4d} segments, distance {dist_text}"
            )

    # ------------------------------------------------------------------
    # Spine and synapse helpers used by bAPs.py.
    # ------------------------------------------------------------------

    def _prepare_spine_targets(self, min_distance_um=60):
        """Record dendritic segments where explicit spines may be added.

        The model is kept simpler near the soma. Distal basal/apical dendrites
        farther than ``min_distance_um`` can receive passive spine compartments.
        """

        self.spine_segments = self.segments_in_distance_range("basal", min_distance_um)
        self.spine_segments += self.segments_in_distance_range("apical", min_distance_um)
        self.segment_to_spines = {self._segment_key(seg): [] for seg in self.spine_segments}
        self.segment_to_next_spine = {self._segment_key(seg): 0 for seg in self.spine_segments}

    def _segment_key(self, seg):
        """Stable key for a segment.

        NEURON may create fresh Python Segment wrapper objects for the same
        biological segment, so we key by section name and x location instead of
        relying on Python object identity.
        """

        return (seg.sec.name(), round(float(seg.x), 10))

    def add_single_spine(
        self,
        seg,
        neck_d=0.25,
        neck_l=1.35,
        head_area=2.8,
        RA=261.97,
        offset=False,
    ):
        """Attach one passive spine to ``seg`` and return ``(neck, head)``.

        A spine is represented by two sections:
        * neck: thin connector from dendrite to spine head;
        * head: small passive compartment where synapses can be placed.
        """

        target_seg = self._random_location_inside_segment(seg) if offset else seg

        neck = h.Section(name=f"{self.HCell}.spine_neck[{self.spine_num}]")
        neck.L = neck_l
        neck.diam = neck_d
        neck.nseg = 1
        self._set_passive_spine_properties(neck, RA)

        head_l = 2 * np.sqrt(head_area / (4 * np.pi))
        head = h.Section(name=f"{self.HCell}.spine_head[{self.spine_num}]")
        head.L = head_l
        head.diam = head_l
        head.nseg = 1
        self._set_passive_spine_properties(head, RA)

        head.connect(neck(1), 0)
        neck.connect(target_seg, 0)

        key = self._segment_key(seg)
        if key in self.segment_to_spines:
            self.segment_to_spines[key].append((neck, head))
        self.spines.append((neck, head))
        self.spine_num += 1
        return neck, head

    def _random_location_inside_segment(self, seg):
        """Choose a nearby x value that stays inside the same computational segment."""

        half_width = 0.5 / seg.sec.nseg
        x_min = max(0, seg.x - half_width + 1e-6)
        x_max = min(1, seg.x + half_width - 1e-6)
        return seg.sec(self.rng.uniform(x_min, x_max))

    def _set_passive_spine_properties(self, sec, RA):
        """Give a spine neck/head simple passive membrane properties."""

        sec.insert("pas")
        sec.g_pas = 1.0 / self.RM
        sec.e_pas = self.E_PAS
        sec.Ra = RA
        sec.cm = self.CM

    def add_full_spine(self, HCell=None, neck_d=0.25, neck_l=1.35, head_area=2.8, RA=261.97):
        """Populate distal dendritic segments with passive spines.

        ``HCell`` is accepted for compatibility with the older assignment code;
        the method always uses ``self.HCell``.
        """

        if not self.spine_segments:
            self._prepare_spine_targets(min_distance_um=60)

        for seg in self.spine_segments:
            n_spines = int(seg.sec.L / seg.sec.nseg * self.spine_density)
            key = self._segment_key(seg)
            existing = len(self.segment_to_spines.get(key, []))
            for _ in range(max(0, n_spines - existing)):
                self.add_single_spine(seg, neck_d, neck_l, head_area, RA, offset=True)

        print(f"{self.spine_num} spines")

    def stim_on_spine(self, seg, neck_d=0.25, neck_l=1.35, head_area=2.8, RA=261.97):
        """Return a spine head for stimulation, creating one if needed.

        If ``seg`` is not in the distal spine-eligible set, return the dendritic
        segment itself. This keeps proximal stimulation simple.
        """

        key = self._segment_key(seg)
        if key not in self.segment_to_spines:
            if not self._is_spine_eligible(seg):
                return None, seg
            self.segment_to_spines[key] = []
            self.segment_to_next_spine[key] = 0

        spines = self.segment_to_spines[key]
        next_spine = self.segment_to_next_spine.get(key, 0)
        if next_spine < len(spines):
            neck, head = spines[next_spine]
        else:
            neck, head = self.add_single_spine(seg, neck_d, neck_l, head_area, RA)
        self.segment_to_next_spine[key] = next_spine + 1
        return neck, head

    def _is_spine_eligible(self, seg):
        """Return True for distal dendritic segments that may receive spines."""

        sec_name = seg.sec.name()
        is_dendrite = ".dend[" in sec_name or ".apic[" in sec_name
        return is_dendrite and self.distance_from_soma(seg) >= 60

    def _as_segment(self, target, x):
        """Return ``target`` as a Segment.

        ``stim_on_spine`` returns either a dendritic Segment or a spine-head
        Section. Synapses need a Segment, so a Section is converted with
        ``section(x)``.
        """

        if hasattr(target, "sec") and hasattr(target, "x"):
            return target
        return target(x)

    def fill_synapse_list_with_spine(self, seglist, config):
        """Move requested synaptic targets onto spine heads when possible."""

        synaptic_segments = []
        for seg in seglist:
            _, target = self.stim_on_spine(seg)
            synaptic_segments.append(self._as_segment(target, config.SPINE_HEAD_X))
        return synaptic_segments

    def fill_clustered_synapses_list_with_spine(self, seglist, rd, config):
        """Create clustered synaptic target segments around each input segment.

        For each center segment in ``seglist``, choose ``config.CLUSTER_SIZE``
        locations within ``config.CLUSTER_L`` um along the same section. Distal
        dendritic targets are moved onto spine heads when possible.
        """

        synaptic_segments = []
        for center_seg in seglist:
            sec = center_seg.sec
            for _ in range(config.CLUSTER_SIZE):
                x = self._jitter_x_within_section(sec, center_seg.x, rd, config.CLUSTER_L)
                _, target = self.stim_on_spine(sec(x))
                synaptic_segments.append(self._as_segment(target, config.SPINE_HEAD_X))
        return synaptic_segments

    def _jitter_x_within_section(self, sec, center_x, rd, cluster_l_um):
        """Choose an x location within ``cluster_l_um`` on one section."""

        if sec.L <= cluster_l_um:
            return rd.uniform(0, 1)

        half_cluster_x = cluster_l_um / (2 * sec.L)
        x_min = max(0, center_x - half_cluster_x)
        x_max = min(1, center_x + half_cluster_x)
        return rd.uniform(x_min, x_max)

    def add_synapses_on_list_of_segments(self, list_of_segments, syn_list, con_list, config):
        """Place paired AMPA and NMDA synapses on each target segment.

        The first pass creates all AMPA synapses; the second pass creates all
        NMDA synapses. This ordering matches ``configure_synaptic_delayes``.
        """

        for seg in list_of_segments:
            ampa = h.Exp2Syn(seg.x, sec=seg.sec)
            ampa.e = config.E_SYN
            ampa.tau1 = config.TAU_1_AMPA
            ampa.tau2 = config.TAU_2_AMPA
            nc = h.NetCon(config.stim, ampa)
            nc.weight[0] = config.AMPA_W
            syn_list.append(ampa)
            con_list.append(nc)

        for seg in list_of_segments:
            nmda = h.NMDA(seg.x, sec=seg.sec)
            nmda.e = config.E_SYN
            nmda.tau_r_NMDA = config.TAU_1_NMDA
            nmda.tau_d_NMDA = config.TAU_2_NMDA
            nmda.n_NMDA = config.N_NMDA
            nmda.gama_NMDA = config.GAMMA_NMDA
            nc = h.NetCon(config.stim, nmda)
            nc.weight[0] = config.NMDA_W
            syn_list.append(nmda)
            con_list.append(nc)
