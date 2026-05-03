"""Structural fuzz harness for the autobio plugin.

Runs generated or scripted scenarios against a real AutobioContextEngine
(real LLM, configurable model) and checks structural invariants after
every step. Transition points (fold, merge, pin retire, speculative
commit) are dumped to a markdown report for hand spot-check.
"""
