# render_peptide.pml
# Load the structure (PDB filename is passed as an argument)
load $1
hide all
show cartoon
color spectrum
set ray_shadows, 0
bg_color white
zoom
rotate y, 90
ray 1200,1200
png $2
quit