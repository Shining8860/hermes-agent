"""Offline cooldown regression from t_d70b399d, using real imports instead of AST extraction.

Synthetic credentials and a temporary auth store drive the real policy/pool/resolver chain.
No live login, API request, profile config, or cooldown is touched.
"""
import json
import time
from unittest.mock import Mock

import pytest

from tests.gateway.test_managed_worker_route_key import _execute, _with_turn_chain
from tests.gateway.test_managed_turn_contract import assignment

MODEL = 'claude-opus-5-5'
PRIMARY = 'synthetic-primary-credential'
FALLBACK = 'synthetic-fallback-credential'
CHAIN = [{'provider': 'deepseek', 'model': 'deepseek-flash'}]


@pytest.fixture(autouse=True)
def no_transport(monkeypatch):
    calls = []
    def refused(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Offline regression attempted HTTP transport')
    monkeypatch.setattr('httpx.Client.send', refused)
    monkeypatch.setattr('httpx.AsyncClient.send', refused)
    yield
    assert calls == []


@pytest.fixture
def route(tmp_path, monkeypatch):
    from agent import anthropic_credentials
    home = tmp_path / 'route-home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('CLAUDE_CONFIG_DIR', str(tmp_path / 'no-claude'))
    monkeypatch.setenv('ANTHROPIC_API_KEY', PRIMARY)
    monkeypatch.setenv('DEEPSEEK_API_KEY', FALLBACK)
    monkeypatch.setattr(anthropic_credentials, 'read_claude_code_credentials', lambda: None)
    frame, _ = assignment(tmp_path, monkeypatch)
    config = json.loads(frame['policy']['config_json'])
    config['model'] = {'default': MODEL, 'provider': 'anthropic'}
    policy = {**frame['policy'], 'model': MODEL, 'config_json': json.dumps(config), 'request_json': '{}'}
    frame = _with_turn_chain({**frame, 'policy': policy, 'api_key': None}, CHAIN)

    def cooldown(enabled=True):
        from agent.credential_pool import load_pool
        (home / 'auth.json').write_text(json.dumps({'credential_pool': {}}))
        pool = load_pool('anthropic')
        if enabled:
            pool.mark_exhausted_and_rotate(status_code=429, api_key_hint=PRIMARY,
                failure_reason='rate_limit', model=MODEL, error_context={'reset_at': time.time() + 600})
        assert (load_pool('anthropic').select(model=MODEL) is None) == enabled
    return frame, cooldown, home


def test_primary_and_model_cooldown_adopt_only_the_resolved_route(route, monkeypatch):
    """The diagnostic control + failure case: a model-less token exists during Opus cooldown."""
    from agent.anthropic_credentials import resolve_anthropic_token
    frame, cooldown, _ = route
    cooldown(False)
    primary = _execute(frame, monkeypatch)
    assert primary['model'] == MODEL and primary['provider'] == 'anthropic'
    assert primary['api_key'] == PRIMARY
    assert primary['api_mode'] == 'anthropic_messages'
    assert primary['base_url'] == 'https://api.anthropic.com'
    assert primary['credential_pool'].provider == 'anthropic'

    cooldown()
    assert resolve_anthropic_token() == PRIMARY
    assert resolve_anthropic_token(model=MODEL) is None
    fallback = _execute(frame, monkeypatch)
    assert fallback['model'] == 'deepseek-flash'
    assert fallback['provider'] == 'deepseek'
    assert fallback['base_url'].startswith('https://api.deepseek.com')
    assert fallback['api_key'] == FALLBACK
    assert fallback['api_mode'] == 'chat_completions'
    assert fallback['credential_pool'].provider == 'deepseek'
    assert fallback['fallback_model'] == CHAIN
    # The next attempt resolves afresh; no policy mutation pins it to the fallback forever.
    cooldown(False)
    recovered = _execute(frame, monkeypatch)
    assert recovered['model'] == MODEL and recovered['api_key'] == PRIMARY


@pytest.mark.parametrize('chain', [[], [{'provider': 'anthropic', 'model': MODEL}]])
def test_exhausted_chain_is_typed_quota_and_books_retryable_exit(route, monkeypatch, chain):
    from agent.managed_worker import _construct_agent
    from gateway import session_kanban
    from gateway.session_policy import restore_policy
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error
    from hermes_cli.kanban_db_dispatch import _exit_code_kind
    frame, cooldown, _ = route
    cooldown()
    frame = _with_turn_chain(frame, chain)
    frame['policy']['kanban_json'] = json.dumps({'synthetic': True})
    records = []
    monkeypatch.setattr(session_kanban, '_record_worker_result', lambda context, code, output: records.append(code))
    policy = restore_policy(frame['policy'])
    factory = Mock(side_effect=AssertionError('Construction must be refused during cooldown'))
    with pytest.raises(AuthError) as raised:
        _construct_agent(frame, factory, policy,
                         model=MODEL, provider='anthropic', api_key=None,
                         fallback_model=chain)
    factory.assert_not_called()
    assert is_rate_limited_auth_error(raised.value)
    assert raised.value.provider == 'anthropic' and raised.value.retryable is True
    assert records == [75]
    assert _exit_code_kind(records[0]) == ('rate_limited', 75)


def test_fallback_recomputes_fast_overrides_and_preserves_provider_body(route, monkeypatch):
    from gateway import session_worker_construct as construct
    frame, _, _ = route
    config = json.loads(frame['policy']['config_json'])
    config.setdefault('agent', {})['service_tier'] = 'priority'
    frame['policy']['config_json'] = json.dumps(config)
    provider_body = {'extra_body': {'fixture': True}}
    monkeypatch.setattr(construct, 'route_runtime', lambda *a: (
        {'provider': 'openai-codex', 'api_key': FALLBACK, 'base_url': 'https://chatgpt.com/backend-api/codex',
         'api_mode': 'codex_responses', 'request_overrides': provider_body},
        {'provider': 'openai-codex', 'model': 'fallback-model'}))
    calls = []
    def fast(model, **kwargs):
        calls.append((model, kwargs['provider']))
        return {'service_tier': 'priority'} if model == 'fallback-model' else {'wrong-primary-route': True}
    monkeypatch.setattr('hermes_cli.models.resolve_fast_mode_overrides', fast)
    built = _execute(frame, monkeypatch)
    assert calls[-1] == ('fallback-model', 'openai-codex')
    assert built['request_overrides'] == {'extra_body': {'fixture': True}, 'service_tier': 'priority'}
    assert provider_body == {'extra_body': {'fixture': True}}


def test_fallback_overwrites_wire_pool_and_model_specific_overrides(route, monkeypatch):
    from gateway import session_worker_construct as construct
    frame, _, _ = route
    pool = object()
    runtime = {'api_key': FALLBACK, 'provider': 'openai-codex', 'requested_provider': 'openai-codex',
               'base_url': 'https://chatgpt.com/backend-api/codex', 'api_mode': 'codex_responses',
               'credential_pool': pool, 'request_overrides': {'reasoning': {'effort': 'low'}},
               'command': 'fixture-command', 'args': ['fixture-arg']}
    monkeypatch.setattr(construct, 'route_runtime', lambda *a: (runtime, {'provider': 'openai-codex', 'model': 'fallback-model'}))
    built = _execute(frame, monkeypatch)
    assert built['model'] == 'fallback-model'
    for key, value in runtime.items():
        assert built[key] == value


def test_fallback_key_reaches_only_the_fallback_sdk(route, monkeypatch):
    """The former offline SDK probe now proves that no empty primary client is constructed."""
    from run_agent import AIAgent
    from agent.anthropic_adapter import build_anthropic_client
    frame, cooldown, _ = route
    cooldown()
    primary_builder = Mock(wraps=build_anthropic_client)
    monkeypatch.setattr('agent.anthropic_adapter.build_anthropic_client', primary_builder)
    clients = []
    class Agent(AIAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            clients.append(self)
    _execute(frame, monkeypatch, base=Agent)
    assert not primary_builder.called
    assert clients and clients[0].provider == 'deepseek'
    client = clients[0].client
    assert client.api_key == FALLBACK
    assert 'api.deepseek.com' in str(client.base_url)
    client.close()


def test_fallback_resolution_is_profile_scoped(route, monkeypatch, tmp_path):
    from agent.secret_scope import set_secret_scope, reset_secret_scope
    from agent.credential_pool import load_pool
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    frame, _, _ = route
    snapshots = []
    # Two profiles with opposing credentials/cooldowns share a process, never a route.
    for label, blocked in [('secondary', True), ('default', False), ('secondary', True)]:
        home = tmp_path / label
        home.mkdir(exist_ok=True)
        own_primary, own_fallback = f'{PRIMARY}-{label}', f'{FALLBACK}-{label}'
        monkeypatch.setenv('HERMES_HOME', str(home))
        monkeypatch.setenv('ANTHROPIC_API_KEY', own_primary)
        monkeypatch.setenv('DEEPSEEK_API_KEY', own_fallback)
        token = set_hermes_home_override(home)
        scope = set_secret_scope({'ANTHROPIC_API_KEY': own_primary, 'DEEPSEEK_API_KEY': own_fallback}, profile_home=str(home))
        try:
            (home / 'auth.json').write_text(json.dumps({'credential_pool': {}}))
            if blocked:
                load_pool('anthropic').mark_exhausted_and_rotate(status_code=429, api_key_hint=own_primary,
                    failure_reason='rate_limit', model=MODEL, error_context={'reset_at': time.time() + 600})
                assert load_pool('anthropic').select(model=MODEL) is None
                from agent.anthropic_credentials import resolve_anthropic_token
                assert resolve_anthropic_token(model=MODEL) is None
            built = _execute(frame, monkeypatch)
            snapshots.append((built['provider'], built['api_key']))
        finally:
            reset_secret_scope(scope)
            reset_hermes_home_override(token)
    assert snapshots == [('deepseek', FALLBACK + '-secondary'), ('anthropic', PRIMARY + '-default'),
                         ('deepseek', FALLBACK + '-secondary')]
