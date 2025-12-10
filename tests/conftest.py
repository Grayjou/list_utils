

import pytest


def pytest_sessionfinish(session, exitstatus):
    """Write failed test reports to reports.txt after test session."""
    if exitstatus == 0:
        return  # No failures
    
    failures = []
    
    # Iterate through all test reports
    for item in session.items:
        # Get the test report for this item
        report = item.stash.get(pytest.StashKey(), None)
        if report and report.failed:
            failures.append(f"{report.nodeid}\n{report.longrepr}\n\n")
    
    # Fallback: use terminal reporter if available
    if not failures and hasattr(session.config, 'pluginmanager'):
        terminalreporter = session.config.pluginmanager.get_plugin('terminalreporter')
        if terminalreporter:
            for rep in terminalreporter.stats.get('failed', []):
                failures.append(f"{rep.nodeid}\n{rep.longrepr}\n\n")
    
    if failures:
        with open("list_utils/reports.txt", "w", encoding="utf-8") as f:
            f.writelines(failures)
