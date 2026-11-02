def portfolio_result(data, target_name, probability_name, balance_name, spend_name, threshold):
    """Default rate and 12-month revenue for accounts with PD below the threshold."""
    accepted = data.loc[data[probability_name] < threshold]
    if len(accepted) == 0:
        return float("nan"), 0.0

    default_rate = float(accepted[target_name].mean())
    monthly = 0.02 * accepted[balance_name] + 0.001 * accepted[spend_name]
    revenue = 12 * monthly
    # A defaulted account stays in the default rate and contributes no revenue.
    revenue = revenue.where(accepted[target_name] == 0, 0.0)
    return default_rate, float(revenue.sum())
