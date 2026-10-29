"""Permissões do workflow que publica a documentação.

O docs.yml só roda em push na main e em workflow_dispatch, então nenhum check
de PR o executa: sem este teste, um PR que mexa nas permissões passa verde, e o
deploy só falha, ou ganha escrita que não precisa, depois do merge.
"""

from pathlib import Path

import yaml

DOCS_YML = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "docs.yml"


def _workflow():
    return yaml.safe_load(DOCS_YML.read_text(encoding="utf-8"))


def _escopos_com_escrita(permissoes):
    # O bloco permissions aceita o mapa escopo -> nível ou um atalho em texto
    # ("read-all", "write-all"); o atalho de escrita conta como todos os escopos.
    if isinstance(permissoes, str):
        return ["write-all"] if permissoes == "write-all" else []
    return sorted(escopo for escopo, nivel in (permissoes or {}).items() if nivel == "write")


def test_workflow_so_concede_leitura_do_conteudo():
    assert _workflow().get("permissions") == {"contents": "read"}


def test_job_deploy_tem_exatamente_escrita_no_pages_e_token_oidc():
    deploy = _workflow()["jobs"]["deploy"]

    assert deploy.get("permissions") == {"pages": "write", "id-token": "write"}


def test_nenhum_job_alem_do_deploy_escreve():
    jobs = _workflow()["jobs"]
    # Sem o build na lista, o laço abaixo poderia passar sem conferir nada.
    assert "build" in jobs

    escritas = {
        nome: escopos
        for nome, job in jobs.items()
        if nome != "deploy" and (escopos := _escopos_com_escrita(job.get("permissions")))
    }

    assert escritas == {}
