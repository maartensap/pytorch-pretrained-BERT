# IPython log file
import pandas as pd
from IPython import embed
from random import shuffle
import json
raw_df = pd.read_csv("/data/socialComSenseQA/candsForFiltering/predictions.csv",index_col=0)
print(raw_df.shape)
labels = ["xNeed","xIntent","xAttr","xReact","xWant","xEffect","oReact","oWant","oEffect"]

for l in labels:
  raw_df.loc[raw_df["label"] == l,"prob_label"] = raw_df.loc[raw_df["label"] == l,"prob_"+l]
cols = ["text_b","prob_label","pred_label"]

#### Exporting to HTML

## Loading HTML templates

baseHTML = open("exampleSCSQA_template.html","rt").read()
toAdd = ""
divHTML = """      <div style="margin-top: 40px;" class="col col-6 {dim}">
<h5>"{event}"</h5>&nbsp;&mdash;&nbsp;{question}?<br>
<a href="#rawInfo{eid}" data-toggle="collapse">[more info]</a><div id="rawInfo{eid}" class="collapse">
<pre style="font-size: 8pt;"><code>{rawevent}</code></pre>
</div>
{inferenceHTML}
</div>
"""

inferenceHTML = """<div class="qline btn-group-vertical" id="{qid}BtnGrp{eid}">
{inferences}
</div>"""

labelHTML = """<label class="btn zero" for="{id}_{eid}"><input id="{id}_{eid}" name="{qid}" type="checkbox" value="{value}" {checked}/>{value} <small>({score}&nbsp;&mdash;&nbsp;{source})</small></label>"""
headerInfo = """
<p>SocialIQa data scored by BERT-base pretrained on ATOMIC, as a <code>(event,inference) \\in {xNeed, xReact, ..., oEffect}</code> 9-way classification.
</p>
"""

## Adding data
g = raw_df.groupby(by=["text_a","label"])
g = [(ix,group.drop_duplicates(subset=cols)) for ix, group in g]
shuffle(g)
i = 0
dim_cntr = {}
for ix, group in g[:2000]:
  # To-do: missing the questions from the original data. Using the dimension instead
  event, dim = ix
  inferences = group[cols].drop_duplicates().sort_values(by="prob_label",ascending=False).rename(columns={"pred_label":"BERT_pred_label"}).round(4).to_html(index=False)
  infHTML = inferenceHTML.format(inferences=inferences,qid=f"{i:07d}",eid=f"{i:07d}")
  rawevent = json.dumps(group.iloc[:,2:].round(3).set_index("text_b").T.to_dict(),indent=2)
  toAdd += divHTML.format(event=event,question=dim,eid=f"{i:07d}",rawevent=rawevent,inferenceHTML=infHTML,dim=dim)
  i +=1
  dim_cntr[dim] = dim_cntr.get(dim,0)+1


ckboxes = ['<label><input type="checkbox" value="{dim}" checked="checked" onclick="toggleDimension(this);">{dim} ({cnt})</label>'.format(
  dim=dim,cnt=dim_cntr.get(dim,0))
           for dim in labels]
headerInfo += '<div class="col-12">'+"&nbsp;".join(ckboxes)+'</div>'

with open("qaBERTscored2.html","wt+") as f:
  exportHTML = baseHTML.replace("INSERT_HERE_PYTHON", toAdd)
  exportHTML = exportHTML.replace("<!--description header-->", headerInfo)
  f.write(exportHTML)
# embed()
