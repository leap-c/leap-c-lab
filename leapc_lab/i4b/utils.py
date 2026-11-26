"""Helpers shared by the i4b OCP, planner and tests."""

from i4b.constants import C_WATER_SPEC


def thermal_power_kw(mdot_hp, T_supply, T_return):
    """Thermal power the heat pump delivers to the water circuit [kW].

    ``Qth = mdot_hp * c_water * (T_supply - T_return) / 1000``. Works on floats,
    numpy arrays and CasADi symbols alike.

    Args:
        mdot_hp: Mass-flow rate of the HP circuit [kg/s].
        T_supply: Supply temperature [degC].
        T_return: Return temperature [degC].
    """
    return mdot_hp * C_WATER_SPEC * (T_supply - T_return) / 1000
