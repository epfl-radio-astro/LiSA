
class CNN:

    @classmethod
    def make_training_set(domain, truthfilepath, outdir):
        sourcesTruth = TruthSource.catalog_to_sources_in_domain(truthfilepath, domain)
        for s in sourcesTruth:
            cutout = data_value[s.z()-wf:s.z()+wf, s.y()-wcl:s.y()+wcl, s.x()-wcl:s.x()+wcl]
            np.save(save_path+'/cube_'+str(counter))
            counter+=1


'''

sourcesTruth = [TruthSource(s, w = coord) for s in TruthSource.catalog_to_array(save_path+'/sources.txt')]
counter=0
for s in sourcesTruth:
    cutout = data_value[s.z()-wf:s.z()+wf, s.y()-wcl:s.y()+wcl, s.x()-wcl:s.x()+wcl]
    np.save(save_path+'/cube_'+str(counter))
    counter+=1
'''