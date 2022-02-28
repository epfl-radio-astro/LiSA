def make_combined_file(inname, outname):
    count = 0
    file_count = 0
    with open(outname, 'w') as fout:
        fout.write("id ra dec hi_size line_flux_integral central_freq pa i w20\n")
        import glob
        for fname in glob.glob(inname):
            with open(fname, 'r') as fin:
                file_count += 1
                for line in fin:
                    fout.write("{0} {1}".format(count,line))
                    count += 1
    print("Read {0} sources from {1} files, written to {2}".format(count, file_count, outname))

if __name__ == "__main__":

    catalog_filepath = "/scratch/etolley/SDC2/full_output_3Ddenoise_fixed/catalog_new2_th12*.txt"
    output_combinedfile = "/scratch/etolley/SDC2/full_output_3Ddenoise_fixed/combined_catalog_new2.txt"
    make_combined_file(catalog_filepath, output_combinedfile)

    #catalog_filepath = "/scratch/etolley/SDC2/ldev_output_3Ddenoise/catalog_new3_th13*_c0p85*"
    #output_combinedfile = "/scratch/etolley/SDC2/ldev_output_3Ddenoise/combined_catalog3_th13c0p85.txt"
    #make_combined_file(catalog_filepath, output_combinedfile) 
    

# sdc2-score create /scratch/etolley/SDC2/full_output_3Ddenoise_fixed/combined_catalog_new2.txt -u epfl -p "iKe0@yBv\`'s6yr" 
#sdc2-score create /scratch/etolley/SDC2/ldev_output_3Ddenoise/combined_catalog2_th13c0p85.txt -u epfl -p "iKe0@yBv\`'s6yr" -c 1.ldev





