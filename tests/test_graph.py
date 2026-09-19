from arxiv_agent.graph import build_ingestion_graph, build_qa_graph, checkpointer_context


def test_ingestion_graph_compiles_with_expected_nodes():
    graph = build_ingestion_graph()
    nodes = set(graph.get_graph().nodes.keys())
    for expected in [
        "understand_query",
        "fetch_by_id",
        "search_papers",
        "select_paper",
        "fetch_parse",
        "chunk_embed",
        "summarize",
    ]:
        assert expected in nodes


def test_qa_graph_compiles_with_expected_nodes_and_checkpointer(tmp_path, monkeypatch):
    monkeypatch.setattr("arxiv_agent.config.CHECKPOINT_DB_PATH", tmp_path / "test.sqlite")
    monkeypatch.setattr("arxiv_agent.graph.CHECKPOINT_DB_PATH", tmp_path / "test.sqlite")

    with checkpointer_context() as checkpointer:
        graph = build_qa_graph(checkpointer)
        nodes = set(graph.get_graph().nodes.keys())
        for expected in [
            "retrieve_chunks",
            "answer_question",
            "verify_answer",
            "prepare_regeneration",
            "refuse",
            "record_turn",
        ]:
            assert expected in nodes
