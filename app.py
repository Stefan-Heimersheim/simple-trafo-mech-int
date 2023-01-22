from flask import Flask, render_template, request, make_response
from transformer_lens import HookedTransformer, utils
from io import StringIO
import sys
import torch
from functools import partial
import plotly.offline as pyo
import plotly.graph_objs as go

model = HookedTransformer.from_pretrained("gelu-2l")

app = Flask(__name__)




@app.route("/")
def index():
    return render_template("index.html")

def the_model(prompt, answer):
    logits = model(prompt)[0,-1]
    answer_index = logits.argmax()
    answer = model.tokenizer.decode(answer_index)
    return str(answer)

def test_prompt(prompt, answer):
    output = StringIO()
    sys.stdout = output
    utils.test_prompt(prompt, answer, model)
    output = output.getvalue()
    output = output.replace("\n", "<br>")
    return make_response(output)

def patch_stream(clean_prompt=None, answer=None, corrupt_prompt=None, corrupt_answer=None):
    print("clean_prompt", clean_prompt)
    print("answer", answer)
    print("corrupt_prompt", corrupt_prompt)
    print("corrupt_answer", corrupt_answer)
    clean_answer_index = model.tokenizer.encode(answer)[0]
    corrupt_answer_index = model.tokenizer.encode(corrupt_answer)[0]
    _, corrupt_cache = model.run_with_cache(corrupt_prompt)
    clean_tokens = model.to_str_tokens(clean_prompt)
    def patch_residual_stream(activations, hook, layer="blocks.6.hook_resid_post", pos=5):
        activations[:, pos, :] = corrupt_cache[layer][:, pos, :]
        return activations
    layers = ["blocks.0.hook_resid_pre", *[f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]]
    n_layers = len(layers)
    n_pos = len(clean_tokens)
    patching_effect = torch.zeros(n_layers, n_pos)
    for l, layer in enumerate(layers):
        for pos in range(n_pos):
            fwd_hooks = [(layer, partial(patch_residual_stream, layer=layer, pos=pos))]
            prediction_logits = model.run_with_hooks(clean_prompt, fwd_hooks=fwd_hooks)[0, -1]
            patching_effect[l, pos] = prediction_logits[clean_answer_index] - prediction_logits[corrupt_answer_index]
    return patching_effect.detach().numpy()



@app.route("/run_the_model", methods=["POST"])
def run_the_model():
    param1 = request.form["param1"]
    param2 = request.form["param2"]
    # Run the Python code here
    result = the_model(param1,param2)
    # Return the result to the user
    return result

@app.route("/run_test_prompt", methods=["POST"])
def run_test_prompt():
    param1 = request.form["param1"]
    param2 = request.form["param2"]
    # Run the Python code here
    result = test_prompt(param1,param2)
    # Return the result to the user
    return result

@app.route("/run_stream_patch", methods=["POST"])
def run_stream_patch():
    param1 = str(request.form["param1"])
    param2 = str(request.form["param2"])
    param3 = str(request.form["param3"])
    param4 = str(request.form["param4"])
    # Plot
    clean_tokens = model.to_str_tokens(param1)
    token_labels = [f"(pos {i:2}) {t}" for i, t in enumerate(clean_tokens)]
    layers = ["blocks.0.hook_resid_pre", *[f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]]
    patching_effect = patch_stream(clean_prompt=param1, answer=param2, corrupt_prompt=param3, corrupt_answer=param4)
    #imshow(patching_effect, xticks=token_labels, yticks=layers, xlabel="pos", ylabel="layer",
    #   zlabel="Logit difference", title="Patching with 1st occurrence of first name", width=600, height=380)
    data = [go.Heatmap(z=patching_effect, x=token_labels, y=layers)]
    layout = go.Layout(title="My Plot")
    fig = go.Figure(data=data, layout=layout)
    plot_div = pyo.plot(fig, output_type='div', include_plotlyjs=True)
    return plot_div

if __name__ == "__main__":
    app.run(debug=True)

