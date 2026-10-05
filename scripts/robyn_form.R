# Robyn's own adstock and Hill on the cases scripts/robyn_form.py writes: for each, the adstock
# adstock_geometric returns on the spend at the decay, and the inflexion and the curve
# saturation_hill returns on that adstock at the shape and gamma. Run by scripts/robyn_form.py in
# the Robyn arm's scratch library (scripts/robyn_arm.lock.txt), which installs nothing:
#
#   R_LIBS=LIB R_LIBS_USER=LIB timeout 300 Rscript --vanilla scripts/robyn_form.R IN OUT

suppressPackageStartupMessages(library(Robyn))

args <- commandArgs(trailingOnly = TRUE)
cases <- jsonlite::fromJSON(args[1], simplifyVector = FALSE)
read <- lapply(cases, function(case) {
  adstock <- adstock_geometric(as.numeric(unlist(case$spend)), case$decay)$x_decayed
  hill <- saturation_hill(adstock, case$shape, case$gamma)
  list(adstock = adstock, inflexion = hill$inflexion, saturated = hill$x_saturated)
})
jsonlite::write_json(
  list(robyn = as.character(packageVersion("Robyn")), r = R.version.string, cases = read),
  args[2],
  digits = I(17),
  auto_unbox = TRUE
)
