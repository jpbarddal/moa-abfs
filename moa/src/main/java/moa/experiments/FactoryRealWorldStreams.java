/*
 * Copyright (c) 2017.
 * @author Jean Paul Barddal (jean.barddal@ppgia.pucpr.br)
 * @author Heitor Murilo Gomes (heitor.gomes@telecom-paristech.fr)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

package moa.experiments;

import moa.streams.ArffFileStream;
import moa.streams.ExampleStream;
import java.util.HashMap;

public class FactoryRealWorldStreams {
    public final int STREAM_LENGTH = Integer.MAX_VALUE;

    public HashMap<String, ExampleStream> instantiateAll() {
        HashMap<String, ExampleStream> ret = new HashMap<>();
        ret.putAll(instantiateInternetADS());
        ret.putAll(instantiateNOMAO());
        ret.putAll(instantiateSPAM());
        return ret;
    }

    public HashMap<String, ExampleStream> instantiateSPAM() {
        HashMap<String, ExampleStream> ret = new HashMap<>();
        ret.put("SPAM", prepareARFFLoader("./spam.arff"));
        return ret;
    }

    public HashMap<String, ExampleStream> instantiateNOMAO() {
        HashMap<String, ExampleStream> ret = new HashMap<>();
        ret.put("NOMAO", prepareARFFLoader("./nomao.arff"));
        return ret;
    }

    public HashMap<String, ExampleStream> instantiateInternetADS(){
        HashMap<String, ExampleStream> ret = new HashMap<>();
        ret.put("INTERNETADS", prepareARFFLoader("./internet_ads.arff"));
        return ret;
    }

    private ArffFileStream prepareARFFLoader(String path) {
        ArffFileStream arff = new ArffFileStream();
        arff.arffFileOption.setValue(path);
        arff.prepareForUse();
        arff.restart();
        return arff;
    }

}
