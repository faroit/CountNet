import base64
import json
import numpy as np
from werkzeug.wrappers import Request, Response
import predict


def decode_audio(audio_bytes):
    return np.frombuffer(base64.b64decode(audio_bytes), dtype="float32")


def make_app(estimate_func):
    def app(environ, start_response):
        inputs = json.loads(Request(environ).get_data())

        outputs = []
        for inp in inputs:
            try:
                est = int(estimate_func(decode_audio(inp)))
            except Exception as e:
                print(f"Error estimating speaker count for input {len(outputs)}: {e}")
                est = None
            outputs.append(est)

        return Response(json.dumps(outputs))(environ, start_response)

    return app


if __name__ == "__main__":
    import argparse
    import functools
    from werkzeug.serving import run_simple

    parser = argparse.ArgumentParser(
        description="Run simple JSON api server to predict speaker count"
    )
    parser.add_argument("--model", default="CRNN", help="model name")
    args = parser.parse_args()

    model = predict.load_model(args.model)
    scaler = predict.load_scaler()

    app = make_app(functools.partial(predict.count, model=model, scaler=scaler))
    run_simple("0.0.0.0", 5000, app, use_debugger=True)
