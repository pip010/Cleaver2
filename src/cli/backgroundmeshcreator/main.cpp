

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
//
// Cleaver - A MultiMaterial Conforming Tetrahedral Meshing Library
//  - Command Line Program
//
// Primary Author: Jonathan Bronson (bronson@sci.utah.edu)
// Secondary Author: Petar Petrov (pip010@gmail.com)
//
/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
//
//-------------------------------------------------------------------
//
//  Copyright (C) 2014, Jonathan Bronson
//  Scientific Computing & Imaging Institute
//  University of Utah
//
//  Permission is  hereby  granted, free  of charge, to any person
//  obtaining a copy of this software and associated documentation
//  files  ( the "Software" ),  to  deal in  the  Software without
//  restriction, including  without limitation the rights to  use,
//  copy, modify,  merge, publish, distribute, sublicense,  and/or
//  sell copies of the Software, and to permit persons to whom the
//  Software is  furnished  to do  so,  subject  to  the following
//  conditions:
//
//  The above  copyright notice  and  this permission notice shall
//  be included  in  all copies  or  substantial  portions  of the
//  Software.
//
//  THE SOFTWARE IS  PROVIDED  "AS IS",  WITHOUT  WARRANTY  OF ANY
//  KIND,  EXPRESS OR IMPLIED, INCLUDING  BUT NOT  LIMITED  TO THE
//  WARRANTIES   OF  MERCHANTABILITY,  FITNESS  FOR  A  PARTICULAR
//  PURPOSE AND NONINFRINGEMENT. IN NO EVENT  SHALL THE AUTHORS OR
//  COPYRIGHT HOLDERS  BE  LIABLE FOR  ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
//  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//  USE OR OTHER DEALINGS IN THE SOFTWARE.
//-------------------------------------------------------------------
//
/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include <Cleaver/Cleaver.h>
#include <Cleaver/CleaverMesher.h>
#include <Cleaver/InverseField.h>
#include <Cleaver/SizingFieldCreator.h>
#include <nrrd2cleaver/nrrd2cleaver.h>
#include <Cleaver/Timer.h>

#include <boost/program_options.hpp>

// STL Includes
#include <exception>
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <ctime>

const std::string kDefaultOutputName   = "bgmesh";

const double kDefaultAlpha = 0.4;
const double kDefaultAlphaLong = 0.357;
const double kDefaultAlphaShort = 0.203;


namespace po = boost::program_options;

// Entry Point
int main(int argc,	char* argv[])
{
    bool verbose = false;
    bool have_sizing_field = false;
    std::vector<std::string> material_fields;
    std::string output_path = kDefaultOutputName;
    double alpha = kDefaultAlpha;
    double alpha_long = kDefaultAlphaLong;
    double alpha_short = kDefaultAlphaShort;
    std::string sizing_field;
    enum cleaver::MeshType mesh_mode = cleaver::Structured;
    double background_time = 0;

    //-------------------------------
    //  Parse Command Line Params
    //-------------------------------
    try{
        po::options_description description("Command line flags");
        description.add_options()
                ("help,h", "display help message")
                ("verbose,v", "enable verbose output")
                ("version,V", "display version information")
                ("material_fields,i", po::value<std::vector<std::string> >()->multitoken(), "material field paths")
                ("alpha,a", po::value<double>(), "initial alpha value")
                ("alpha_short,s", po::value<double>(), "alpha short value for regular mesh_mode")
                ("alpha_long,l", po::value<double>(), "alpha long value for regular mesh_mode")
                ("mesh_mode,m", po::value<std::string>(), "background mesh mode (structured [default], regular)")
                ("sizing_field,z", po::value<std::string>(), "sizing field path")
                ("output", po::value<std::string>()->default_value(kDefaultOutputName, "bgmesh"), "output path")
        ;

        boost::program_options::variables_map variables_map;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, description), variables_map);
        boost::program_options::notify(variables_map);

        // print version info
        if (variables_map.count("version")) {
            std::cout << cleaver::Version << std::endl;
            return 0;
        }
        // print help
        else if (variables_map.count("help") || (argc ==1)) {
            std::cout << description << std::endl;
            return 0;
        }

        // enable verbose mode
        if (variables_map.count("verbose")) {
            verbose = true;
        }

        //alphas
        if (variables_map.count("alpha")) {
          alpha = variables_map["alpha"].as<double>();
        }
        if (variables_map.count("alpha_short")) {
          alpha_short = variables_map["alpha_short"].as<double>();
        }
        if (variables_map.count("alpha_long")) {
          alpha_long = variables_map["alpha_long"].as<double>();
        }

        // parse the background mesh mode
        if (variables_map.count("mesh_mode")) {
          std::string mesh_mode_string = variables_map["mesh_mode"].as<std::string>();
          if(mesh_mode_string.compare("regular") == 0) {
            mesh_mode = cleaver::Regular;
          }
          else if(mesh_mode_string.compare("structured") == 0) {
            mesh_mode = cleaver::Structured;
          } else {
            std::cerr << "Error: invalid background mesh mode: " << mesh_mode_string << std::endl;
            std::cerr << "Valid Modes: [regular] [structured] " << std::endl;
            return 6;
          }
        }

        // parse the material field input file names
        if (variables_map.count("material_fields")) {
            material_fields = variables_map["material_fields"].as<std::vector<std::string> >();
            int file_count = material_fields.size();
        }
        else{
            std::cout << "Error: At least one material field file must be specified." << std::endl;
            return 0;
        }

        //-----------------------------------------
        // parse the sizing field input file name
        // and NOT check for conflicting parameters
        //----------------------------------------
        if (variables_map.count("sizing_field")) {
          have_sizing_field = true;
          sizing_field = variables_map["sizing_field"].as<std::string>();
        }

        // set output path
        if (variables_map.count("output")) {
            output_path = variables_map["output"].as<std::string>();
        }
    }
    catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 0;
    }
    catch(...) {
        std::cerr << "Unhandled exception caught. Terminating." << std::endl;
        return 0;
    }

    //-----------------------------------
    //  Load Data & Construct  Volume
    //-----------------------------------
    std::cout << " Loading input fields:" << std::endl;
    for (size_t i=0; i < material_fields.size(); i++) {
        std::cout << " - " << material_fields[i] << std::endl;
    }

    std::vector<cleaver::AbstractScalarField*> fields = loadNRRDFiles(material_fields, verbose);
    if(fields.empty()){
        std::cerr << "Failed to load image data. Terminating." << std::endl;
        return 0;
    }
    else if(fields.size() == 1) {
        fields.push_back(new cleaver::InverseScalarField(fields[0]));
    }

    cleaver::Volume *volume = new cleaver::Volume(fields);

    cleaver::CleaverMesher mesher(volume);
    mesher.setAlphaInit(alpha);

    //------------------------------------------------------------
    // Load Sizing Field
    //------------------------------------------------------------
    cleaver::AbstractScalarField *sizingField = NULL;
    if (have_sizing_field)
    {
      std::cout << "Loading sizing field: " << sizing_field << std::endl;
      sizingField = loadNRRDFile(sizing_field, verbose);
    }
    else
    {
      std::cerr << "Sizing Field file required !" << '\n';
      return 2;
    }

    //------------------------------------------------------------
    // Set Sizing Field on Volume
    //------------------------------------------------------------
    volume->setSizingField(sizingField);


    //-----------------------------------------------------------
    // Construct Background Mesh
    //-----------------------------------------------------------
    cleaver::Timer background_timer;
    background_timer.start();

    cleaver::TetMesh *bgMesh = NULL;

    if(verbose)
      std::cout << "Creating Octree Mesh..." << std::endl;

    switch(mesh_mode)
    {
      case cleaver::Regular:
        mesher.setAlphas(alpha_long,alpha_short);
        mesher.setRegular(true);
        bgMesh = mesher.createBackgroundMesh(verbose);
        break;

      default:
      case cleaver::Structured:
        mesher.setRegular(false);
        bgMesh = mesher.createBackgroundMesh(verbose);
        break;
    }

    background_timer.stop();
    background_time = background_timer.time();
    mesher.setBackgroundTime(background_time);

    //-----------------------------------------------------------
    // Write Background Mesh
    //-----------------------------------------------------------
    if (bgMesh ) {
      bgMesh->writeNodeEle(output_path, false, false, false);
    }

    //-----------------------------------------------------------
    // THE END
    //-----------------------------------------------------------
    std::cout << " Done." << std::endl;
    return 0;
}
