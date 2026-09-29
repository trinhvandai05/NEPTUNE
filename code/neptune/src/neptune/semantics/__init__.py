"""Model-INDEPENDENT semantic statistics that define the H1 strata.

Hard rule, enforced by tests/test_semantics_isolation.py: nothing in this
package may import neptune.model, neptune.state, neptune.dynamics,
neptune.heads, neptune.training or neptune.content.encoder.  If the
stratification variable could see the trained model, H1 would be scored on a
partition the model itself produced -- a circular test.
"""
