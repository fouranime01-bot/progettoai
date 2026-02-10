def test_imports():
    import src.model
    import src.training
    import src.predict
    import src.eda
    assert True


def test_model_forward():
    import torch
    from src.model import SimpleNet

    model = SimpleNet()
    x = torch.randn(1, 1, 28, 28)
    out = model(x)

    assert out.shape == (1, 10)
