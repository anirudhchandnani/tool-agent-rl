# Fix advantage normalization in REINFORCE

## Problem
The REINFORCE advantages were scaled by their standard deviation but the batch
baseline was never subtracted first. Dividing raw returns by their spread leaves
a large positive offset, so almost every action gets pushed up and the baseline
does nothing to cut variance. Training was slower and noisier than it should be.

## Fix
Subtract the batch mean before dividing by the standard deviation, so the
advantage is properly centered on the baseline.

## Check
The learning test for REINFORCE is tighter after the fix and reward climbs
faster on a short run.
