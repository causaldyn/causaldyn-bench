"""Family 9, family 8's geo worlds with ring spillover: family 8's and family 10's exports keep
their bits now that a world may spill."""

import pytest

from causaldyn_bench.scorecard.few_geos import FEW_GEOS
from causaldyn_bench.scorecard.geo import GEO

# Each environment's first pilot world's export at rungs 0 and 4, read before ``mix`` took a
# spillover.
DIGESTS = {
    (GEO, "policy-0"): ("b35cccc30d161f74", "6960cc8dcd601dd1"),
    (GEO, "policy-0.5"): ("a846b14a3a7adf90", "7876835bb73ac1b3"),
    (FEW_GEOS, "geos-3"): ("ef143986dfd2274a", "1859cea3dad99599"),
    (FEW_GEOS, "geos-50"): ("93e080fc8fbe9bdc", "06667d86d2629408"),
}


@pytest.mark.parametrize(("family", "name"), list(DIGESTS), ids=[n for _, n in DIGESTS])
def test_family_8s_and_10s_exports_keep_their_bits(family, name):
    world = family.world(name, family.pilots[name][0])
    stamps = tuple(family.observe(world, k).digest() for k in (0, 4))
    assert stamps == DIGESTS[family, name]
