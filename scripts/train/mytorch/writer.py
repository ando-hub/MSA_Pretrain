def write_log(writer, epoch, dic):
    for k, v in dic.items():
        if isinstance(v, dict):
            writer.add_scalars(k, v, epoch)
        else:
            writer.add_scalar(k, v, epoch)


def write_model(writer, model, dataloader):
    model.to('cpu')
    x = iter(dataloader).next()[0]
    writer.add_graph(model, x)
