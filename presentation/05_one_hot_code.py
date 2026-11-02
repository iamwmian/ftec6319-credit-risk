# The 11 categorical fields named in the brief.
categorical = [
    "B_30", "B_38", "D_63", "D_64", "D_66", "D_68",
    "D_114", "D_116", "D_117", "D_120", "D_126",
]

# The development sample drew 17 numeric fields. None of the 11 are among them.
present = [name for name in categorical if name in frame.columns]

# One-hot encoding turns each category into its own 0/1 column.
# With no categorical field in the sample, this adds no columns.
encoded = pd.get_dummies(frame[present], dtype="int")
