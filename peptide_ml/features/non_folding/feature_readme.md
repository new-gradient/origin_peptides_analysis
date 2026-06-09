# Non-Folding Feature ReadMe
## Basic Features


## Chemical Features 
### Overview
Chemical features capture fundamental physicochemical properties of peptides that influence their behavior, stability, and interactions. Relevant for understanding peptide function and predicting their behavior in different environments.
### 1. Isoelectric Point
#### Description

The pH at which a peptide carries no net electrical charge
relevant for understanding peptide behavior in different pH environments

#### Implementation Details

Calculated using pKa values of ionizable groups
Considers contribution of all charged amino acids and terminal groups
Uses BioPython's ProteinAnalysis implementation

#### ML Processing

Raw and ML features are identical (value between 0-14)
Already naturally scaled for ML use
Biologically meaningful scale requires no transformation

### 2. Charge Features
#### Description
Calculates net charge at different physiologically relevant pH values:

pH 2: Highly acidic conditions
pH 7: Physiological conditions
pH 12: Highly basic conditions

#### Implementation Details

What pKa means:

pKa is the pH at which a group is 50% ionized
Above their pKa, acidic groups (-COOH) lose H+ and become negatively charged (-COO-)
Above their pKa, basic groups (-NH2) stay neutral
Below their pKa, acidic groups stay neutral
Below their pKa, basic groups gain H+ and become positively charged (-NH3+)

Firstly 
```python

self.pka_values = {
            'K': 10.5,  # Lysine
            'R': 12.5,  # Arginine
            'H': 6.0,   # Histidine
            'D': 3.9,   # Aspartic acid
            'E': 4.3,   # Glutamic acid
            'C': 8.3,   # Cysteine
            'Y': 10.1,  # Tyrosine
            'N_term': 8.0,  # N-terminus
            'C_term': 3.1   # C-terminus
        }
```

1. These are the only amino acids with ionizable side chains in the physiological pH range:

2. These are commonly accepted "model" values

So in the code:
```python
if pka > ph:  # Basic residues
    charge += count
else:  # Acidic residues
    charge -= count
```
For each amino acid:

If its pKa is higher than the pH we're looking at:

It will be protonated (gain H+)
Adds positive charge
This happens to basic residues (K, R, H) at low pH


If its pKa is lower than the pH:

It will be deprotonated (lose H+)
Adds negative charge
This happens to acidic residues (D, E) at high pH


The terminal groups:

```python
if self.pka_values['N_term'] > ph:
    charge += 1
if self.pka_values['C_term'] > ph:
    charge -= 1
```

Similar logic but only one charge possible per terminus
N-terminus can be +1 or 0
C-terminus can be -1 or 0

So using pKa values to determine whether each amino acid will be charged or not at a given pH, and then summing those charges.

#### ML Processing

Raw features: Absolute charges at each pH
ML-ready features: Charges normalized by sequence length
Rationale: Normalization removes length dependency while preserving charge density information

### 3. Local Charge Distribution
#### Description
Analyzes charge distribution along the peptide sequence using a sliding window approach. This is crucial because:

- Local charge concentrations affect surface properties
- Charge distribution impacts protein-protein interactions
- Charge clusters can affect folding and stability
- Important for membrane interactions and cellular uptake

#### Implementation Details
The sliding window mechanism:
```python
def _calculate_charge_profile(self, sequence: str) -> List[float]:
    profile = []
    half_window = self.window_size // 2
    
    for i in range(len(sequence)):
        start = max(0, i - half_window)
        end = min(len(sequence), i + half_window + 1)
        window = sequence[start:end]
```
Key parameters:

Window size = 5 residues

- Small enough to capture local variations
- Large enough to be meaningful
- Odd number ensures centered window
- Approximates one turn of an α-helix



Charge calculation within window:

```python
pos_count = sum(1 for aa in window if aa in self.charged_pos)
neg_count = sum(1 for aa in window if aa in self.charged_neg)
local_charge = (pos_count - neg_count) / len(window)
```

Where:

- Positive charges: K (Lysine), R (Arginine)
- Negative charges: D (Aspartic acid), E (Glutamic acid)
- Normalized by window length to handle edge effects

The profile generation:

Window slides along sequence
At each position:

Counts positive and negative residues
Calculates net charge density
Normalizes by window size


Builds profile of local charge environment

#### ML Processing
Raw features:

- Complete charge profile (list for each position)
- Maximum local charge (highest concentration)
- Minimum local charge (lowest concentration)
- Charge variation (standard deviation)

ML-ready features:
```python
pythonCopyreturn {
    'max_local_charge': max(profile),
    'min_local_charge': min(profile),
    'charge_variation': np.std(profile),
    'charge_asymmetry': np.mean(profile[:len(profile)//2]) - 
                       np.mean(profile[len(profile)//2:])
}
```
Rationale for ML features:

Max/min local charge:

Captures extreme charge concentrations
Important for interaction sites
Independent of sequence length


Charge variation:

Measures charge distribution uniformity
Higher values indicate charged patches
Lower values indicate even distribution


Charge asymmetry:

Compares N-terminal vs C-terminal regions
Important for directional properties
Relevant for membrane interactions



This dimensionality reduction:

Reduces profile (length N) to 4 features
Preserves key charge distribution characteristics
Makes features comparable across different length peptides
Captures biologically relevant properties

### 4. Hydrophobicity Features
#### Description
Measures how hydrophilic or hydrophobic different parts of the peptide are. This is crucial because:

- Determines how peptide interacts with water
- Affects which parts might stick out vs fold inward
- Important for how peptides might interact with cell membranes

Uses two different measurement scales:

1. Kyte-Doolittle Scale:


- Traditional, widely-used scale
- Good at finding regions that might insert into cell membranes
- Values from -4.5 (really likes water) to +4.5 (really avoids water)


2. Eisenberg Scale:


- Alternative way of measuring
- Better at predicting which parts will be on the surface
- Provides a second opinion to validate patterns

#### Implementation Details
Each amino acid gets two scores (one from each scale):
```python
self.kd_scale = {
    'A':  1.8, 'C':  2.5, 'D': -3.5, 'E': -3.5, 'F':  2.8,  # etc
}
```
Examples:

- High positive values (like I: 4.5, V: 4.2): Really water-fearing
- Low negative values (like K: -3.9, R: -4.5): Really water-loving
- Middle values (like G: -0.4, W: -0.9): In between

The code:

- Looks at each amino acid in sequence
- Assigns it scores from both scales
- Creates a profile of how hydrophobicity changes along peptide
- Also calculates if hydrophobicity follows a pattern (like alternating)

Special calculation (Hydrophobic moment):

```python
def _calculate_hydrophobic_moment(self, hydrophobicity_profile: List[float]) -> float:
    moment = 0
    for i, h in enumerate(hydrophobicity_profile):
        angle = i * 100  # Assuming alpha-helix structure
        moment += h * np.exp(complex(0, np.radians(angle)))
    return abs(moment) / len(hydrophobicity_profile)
```

Measures if hydrophobic amino acids are arranged in a spiral pattern
Important because this pattern is common in peptides that interact with cell membranes - *maybe not important for us though ??? - ask Sara*

### ML Processing
Raw features:

- Full profile of hydrophobicity scores
- Average scores for whole peptide

ML-ready features:

Average hydrophobicity (both scales):

Overall how water-loving/fearing the peptide is


Variation in hydrophobicity:

How much hydrophobicity changes along peptide
High variation might mean distinct regions with different properties


Hydrophobic moment:

Strength of spiral pattern
Normalized by length so comparable between peptides


Rationale:

- Reduces long profiles to key numbers
- Keeps most important properties
- Makes peptides of different lengths comparable

### 5. Amphipathicity Features
#### Description
Measures how well a peptide separates its hydrophilic and hydrophobic parts. Like having a two-sided peptide where:

One side likes water
One side avoids water

This is important because:

- Affects how peptides might fold
- Influences protein-protein interactions

#### Implementation Details
For each position in peptide, looks at surrounding amino acids in a window:
```python
self.hydrophobic = set('AILMFWV')  # Water-fearing amino acids
self.polar = set('STNQY')        # Water-loving amino acids
```
Window size = 11 amino acids because:

- About the length of a common protein structure (α-helix)
- Long enough to see patterns
- Short enough to catch local variations

For each window:

- Counts water-loving amino acids
- Counts water-fearing amino acids

Calculates how well they're separated:

```python
amphipathicity = abs(hydrophobic_count - polar_count) / len(window)
```

- Higher value = better separation
- Lower value = more mixed

#### ML Processing
Raw features:

- Complete profile (score for each position)
- Highest amphipathicity found
- Average amphipathicity

ML-ready features:
```python
return {
    'max_amphipathicity': max(profile),        # Strongest separation
    'mean_amphipathicity': np.mean(profile),   # Overall separation
    'amphipathicity_variation': np.std(profile) # How much it changes
}
```
Why these features:

Maximum value:

- Best separation found
- Might be where peptide interacts with membranes

Mean value:

- Overall tendency to separate
- General behavior indicator

Variation:

- How much separation changes
- Might indicate distinct functional regions

This turns:

- Complex pattern into simple numbers
- Works for any peptide length
- Captures key biological properties

Note: This feature complements hydrophobicity features but looks specifically at the separation pattern rather than just how water-loving/fearing the peptide is.

## Pattern Features

### 1. Residue Properties
#### Description
Looks at fundamental physical properties of the peptide:

Mass characteristics:

- Each amino acid has a specific mass (weight)
- Distribution of these masses tells us about composition
- Important for experimental analysis (like mass spectrometry)


Stability predictions:

- How likely peptide is to break down
- Based on amino acid combinations


Mass-per-charge ratio:

- Relationship between mass and charged amino acids
- Relevant for how peptide moves in electric fields
- Important for experimental separation techniques



#### Implementation Details
Mass values for each amino acid:
```python
self.residue_masses = {
    'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
    'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
    # ... etc
}
```
Note:

- Glycine (G) is smallest: 75.1
- Tryptophan (W) is largest: 204.2
- Most fall between 100-180 Daltons

Mass per charge calculation:
```python
charged_count = (
    sequence.count('K') +  # Lysine
    sequence.count('R') +  # Arginine
    sequence.count('D') +  # Aspartic acid
    sequence.count('E')    # Glutamic acid
)
mass_per_charge = average_mass / charged_count
```

- Divides average mass by number of charged amino acids
- Tells us about density of charges relative to mass
- Special case: if no charges, set to infinity

#### ML Processing
Raw features:
```python
return {
    'residue_masses': masses,          # List of all masses
    'average_mass': np.mean(masses),   # Average mass
    'instability_index': protein.instability_index()
}
```
ML-ready features:
```python
return {
    'average_mass': np.mean(masses),
    'mass_std': np.std(masses),        # How much masses vary
    'instability_index': protein.instability_index(),
    'mass_per_charge': mass_per_charge
}
```
Why these features:

Average mass:

- Overall size indicator
- Normalized for length


Mass variation:

- Mix of small/large amino acids
- Composition diversity


Instability index:

- Below 40: probably stable
- Above 40: might be unstable
- Based on statistical analysis of known proteins


Mass per charge:

- Electrical mobility predictor
- Separation behavior indicator
- Important for experimental methods



These features help predict:

- Physical behavior
- Experimental properties
- Storage stability
- Purification characteristics

