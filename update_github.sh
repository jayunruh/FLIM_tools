cp -R * /Users/jru/Documents/GitHub/napari_jflowcyte/
cd /Users/jru/Documents/GitHub/napari_jflowcyte
rm update_github.*
rm *.bat
git add *
git commit -m "update"
git push --all --repo=https://github.com/jayunruh/napari_jflowcyte.git
cd /Users/jru/Documents/IPython_Notebooks/napari_jflowcyte
