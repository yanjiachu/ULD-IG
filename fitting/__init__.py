from . import power, vanilla, rectified, sld, uld_ig


FITTING_METHODS = {
    "Power": power.FITTING_METHODS,
    "Vanilla": vanilla.FITTING_METHODS,
    "Rectified": rectified.FITTING_METHODS,
    "SLD": sld.FITTING_METHODS,
    "ULD-IG": uld_ig.FITTING_METHODS,
}
